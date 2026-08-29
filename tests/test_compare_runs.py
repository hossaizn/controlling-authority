"""Tests for the paired run comparison.

**Treated as production code, per DL-26.** A Phase 6 review found every eval
scorer could be mutated to return True with the suite still green. This one
decides whether a pre-registered prediction is graded as confirmed or refuted,
which makes it exactly the kind of code that must not be trusted on inspection.
"""

from __future__ import annotations

import pytest

from eval.compare_runs import compare, format_report


def outcome(sid, expected, predicted, slice_name="straightforward"):
    return {
        "scenario_id": sid,
        "slice_name": slice_name,
        "expected": expected,
        "predicted": predicted,
        "expected_fact": None,
        "predicted_fact": None,
    }


def report(outcomes, macro=0.5, version="triage-v3", temperature=None):
    return {
        "macro_accuracy": macro,
        "prompt_version": version,
        "temperature": temperature,
        "outcomes": outcomes,
    }


def test_a_scenario_the_arm_gets_right_and_the_baseline_got_wrong_is_fixed():
    a = report([outcome("s-1", "answer", "refuse")])
    b = report([outcome("s-1", "answer", "answer")])
    result = compare(a, b)
    assert result["counts"]["fixed"] == 1
    assert result["net_scenarios"] == 1


def test_a_scenario_the_arm_breaks_is_counted_against_it():
    a = report([outcome("s-1", "answer", "answer")])
    b = report([outcome("s-1", "answer", "refuse")])
    result = compare(a, b)
    assert result["counts"]["broken"] == 1
    assert result["net_scenarios"] == -1


def test_wrong_in_both_but_differently_is_churn_and_moves_nothing():
    """**The bucket a macro delta cannot show.**

    Sampling variance surfaces here first: the run changed, the score did not.
    Folding this into `agreed` would report an unstable model as a stable one.
    """
    a = report([outcome("s-1", "answer", "refuse")])
    b = report([outcome("s-1", "answer", "clarify")])
    result = compare(a, b)
    assert result["counts"]["churn"] == 1
    assert result["net_scenarios"] == 0
    assert result["changed"] == 1
    assert result["agreement_rate"] == 0.0


def test_wrong_in_both_in_the_same_way_is_agreement_not_churn():
    a = report([outcome("s-1", "answer", "refuse")])
    b = report([outcome("s-1", "answer", "refuse")])
    result = compare(a, b)
    assert result["counts"]["agreed"] == 1
    assert result["changed"] == 0


def test_offsetting_changes_report_a_zero_net_and_a_nonzero_changed():
    """**The failure this module exists to prevent.**

    One fixed and one broken cancel in the average. Reporting only the delta
    would call that "no effect" when the model moved on two scenarios and
    happened to break even.
    """
    a = report([outcome("s-1", "answer", "refuse"), outcome("s-2", "refuse", "refuse")])
    b = report([outcome("s-1", "answer", "answer"), outcome("s-2", "refuse", "answer")])
    result = compare(a, b)
    assert result["net_scenarios"] == 0
    assert result["changed"] == 2
    assert result["counts"] == {"fixed": 1, "broken": 1, "churn": 0, "agreed": 0}


def test_identical_runs_report_full_agreement():
    rows = [outcome("s-1", "answer", "answer"), outcome("s-2", "refuse", "clarify")]
    result = compare(report(rows), report(rows))
    assert result["agreement_rate"] == 1.0
    assert result["changed"] == 0
    assert result["net_scenarios"] == 0


def test_comparing_across_prompt_versions_raises():
    """A prompt edit and a sampling change cannot be separated in one delta, so
    the honest response is to refuse rather than report an uninterpretable
    number. DL-15: a wrong mechanism invalidates the experiment resting on it."""
    a = report([outcome("s-1", "answer", "answer")], version="triage-v2")
    b = report([outcome("s-1", "answer", "answer")], version="triage-v3")
    with pytest.raises(ValueError, match="prompt versions differ"):
        compare(a, b)


def test_comparing_different_scenario_sets_raises():
    """Silently intersecting would score the arm on whichever scenarios it
    happened to complete, which is precisely how DL-24's precedence arm would
    have flattered the open model."""
    a = report([outcome("s-1", "answer", "answer")])
    b = report([outcome("s-2", "answer", "answer")])
    with pytest.raises(ValueError, match="different scenarios"):
        compare(a, b)


def test_the_counts_partition_the_scenario_set():
    """Every scenario lands in exactly one bucket. A scenario counted twice, or
    dropped, would make `net` and `changed` disagree with `n` while every
    individual assertion above still passed."""
    a = report([
        outcome("s-1", "answer", "refuse"),
        outcome("s-2", "answer", "answer"),
        outcome("s-3", "refuse", "answer"),
        outcome("s-4", "clarify", "clarify"),
    ])
    b = report([
        outcome("s-1", "answer", "answer"),
        outcome("s-2", "answer", "refuse"),
        outcome("s-3", "refuse", "clarify"),
        outcome("s-4", "clarify", "clarify"),
    ])
    result = compare(a, b)
    assert sum(result["counts"].values()) == result["n"] == 4
    assert result["counts"] == {"fixed": 1, "broken": 1, "churn": 1, "agreed": 1}


def test_the_report_names_the_provider_default_rather_than_printing_none():
    """`temperature: None` is the shipped configuration, not a missing value.
    Rendering it as "None" reads as a bug in the runner."""
    a = report([outcome("s-1", "answer", "answer")], temperature=None)
    b = report([outcome("s-1", "answer", "answer")], temperature=0.0)
    text = format_report(compare(a, b))
    assert "provider default" in text
    assert "temperature 0" in text
    assert "None" not in text
