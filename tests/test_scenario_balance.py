"""DL-4: the slice balance is fixed before any result is visible.

The headline comparison in this project is agent versus naive baseline. Conflict
cases are exactly where the baseline fails and the agent wins, so a set weighted
toward conflicts would inflate the delta without any engineering behind it.

These targets were committed before a single metric had been computed. Changing
one now breaks a test, which is the point: the mix cannot be quietly adjusted
later once it becomes clear which mix flatters the result.
"""

from __future__ import annotations

from eval.scenarios.loader import load_all, route_counts, slice_counts

# Committed 2026-08-25, before any retrieval or agent code existed.
TARGET_SLICE_COUNTS = {
    "straightforward": 15,
    "ambiguous": 15,
    "control": 10,
    "conflict": 20,
    "superseded": 10,
    "out_of_scope": 10,
    "adversarial": 10,
}


def test_slice_counts_match_the_committed_balance() -> None:
    assert dict(slice_counts()) == TARGET_SLICE_COUNTS


def test_total_is_ninety() -> None:
    assert len(load_all()) == sum(TARGET_SLICE_COUNTS.values()) == 90


def test_conflict_cases_are_a_minority() -> None:
    """The slice the agent is best at must not dominate the headline number."""
    counts = slice_counts()
    assert counts["conflict"] < len(load_all()) / 3


def test_clarify_is_outnumbered_by_answer() -> None:
    """DL-5. If clarifying were the majority behaviour, an agent that always
    asks would score well while being unusable."""
    counts = route_counts()
    assert counts["answer"] > counts["clarify"] * 3


def test_control_slice_is_large_enough_to_punish_over_clarification() -> None:
    """The control slice is the only thing making caution costly. If it is much
    smaller than the ambiguous slice, always-clarify still wins on net."""
    counts = slice_counts()
    assert counts["control"] >= counts["ambiguous"] / 2


def test_every_ambiguous_scenario_names_its_missing_fact() -> None:
    for s in load_all():
        if s.slice == "ambiguous":
            assert s.missing_fact is not None, s.scenario_id


def test_no_control_scenario_expects_clarify() -> None:
    """A control case that expects clarification is a mislabelled ambiguous
    case, and would silently reward the behaviour this slice exists to catch."""
    for s in load_all():
        if s.slice == "control":
            assert s.expected_route != "clarify", s.scenario_id


def test_superseded_scenarios_forbid_the_wrong_version() -> None:
    """A dated question with nothing forbidden can be answered correctly by
    luck, since citing both versions would score as a pass."""
    dated = [
        s
        for s in load_all()
        if s.slice == "superseded" and "LEAVE-004" in " ".join(s.required_citations)
    ]
    unpinned = [s.scenario_id for s in dated if not s.forbidden_citations]
    # superseded-007 asks about the change itself, so both versions are legitimate.
    assert unpinned == ["superseded-007"], unpinned
