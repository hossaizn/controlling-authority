"""Tests for the regression gate itself.

The gate is the thing that will say whether Phase 6 succeeded, so it needs to be
correct before it is trusted. Every prior review in this project found tests that
passed while proving nothing; these assert the gate actually fails when it should.
"""

from __future__ import annotations

from eval.regression import (
    MUST_IMPROVE,
    check,
    format_report,
    load_baseline,
    tolerance_for,
)


def baseline_values() -> dict[str, float]:
    return {k: v["recall@10"] for k, v in load_baseline()["by_slice"].items()}


def test_the_baseline_matches_the_adopted_configuration() -> None:
    """Frozen from voyage-law-2 + structure-aware, the config DL-20 adopted."""
    base = load_baseline()
    assert base["config"]["model"] == "voyage-law-2"
    assert base["config"]["strategy"] == "structure"
    assert base["overall"]["recall@10"] == 0.8947


def test_perfect_slices_get_no_tolerance() -> None:
    """They cannot improve, so any movement is a regression."""
    for name in ("adversarial", "control", "superseded"):
        n = load_baseline()["by_slice"][name]["n"]
        assert tolerance_for(name, n, 1.0) == 0.0


def test_imperfect_slices_get_one_scenario_of_slack_scaled_to_their_size() -> None:
    """One scenario is 5.9 points at n=17 and 10 points at n=10, so a flat
    percentage would be too strict for one slice and too loose for another."""
    assert tolerance_for("straightforward", 17, 0.941) == 1 / 17
    assert tolerance_for("conflict", 18, 0.722) == 1 / 18


def test_holding_the_baseline_passes_every_slice_except_conflict() -> None:
    """Conflict is held to improvement, not to no-regression."""
    verdicts = {v.name: v for v in check(baseline_values())}
    for name in ("adversarial", "control", "superseded", "straightforward"):
        assert verdicts[name].passed, verdicts[name].reason
    assert not verdicts[MUST_IMPROVE].passed
    assert "needs at least" in verdicts[MUST_IMPROVE].reason


def test_conflict_passes_only_when_it_actually_improves() -> None:
    values = baseline_values()
    values["conflict"] = 0.780
    assert next(v for v in check(values) if v.name == "conflict").passed


def test_a_rounding_sized_gain_does_not_count_as_improvement() -> None:
    """The bug the gate had when first run against real data.

    The baseline is stored rounded to four decimals, so an unchanged
    full-precision value tested as greater than it and reported a pass with a
    delta of +0.0 points. Improvement now means at least one more scenario.
    """
    values = baseline_values()
    values["conflict"] = baseline_values()["conflict"] + 1e-5
    verdict = next(v for v in check(values) if v.name == "conflict")
    assert not verdict.passed
    assert "only" in verdict.reason


def test_improvement_must_be_worth_at_least_one_scenario() -> None:
    base = baseline_values()["conflict"]
    just_under = dict(baseline_values(), conflict=base + (1 / 18) - 1e-6)
    just_over = dict(baseline_values(), conflict=base + (1 / 18) + 1e-6)
    assert not next(v for v in check(just_under) if v.name == "conflict").passed
    assert next(v for v in check(just_over) if v.name == "conflict").passed


def test_a_single_scenario_drop_on_a_perfect_slice_fails() -> None:
    """The regression this gate exists to catch: the agent fixes conflict and
    quietly breaks a slice that was already at 1.000."""
    values = baseline_values()
    values["conflict"] = 0.900          # a large, real improvement
    values["superseded"] = 0.900        # one scenario lost at n=10
    verdicts = {v.name: v for v in check(values)}
    assert verdicts["conflict"].passed
    assert not verdicts["superseded"].passed
    assert "REGRESSED" in verdicts["superseded"].reason


def test_a_gain_that_hides_a_loss_is_still_a_failure() -> None:
    """Overall recall is unchanged here, which is exactly why the gate is
    per-slice: conflict gains what straightforward loses."""
    values = baseline_values()
    values["conflict"] = 0.833          # +2 scenarios at n=18
    values["straightforward"] = 0.824   # -2 scenarios at n=17
    verdicts = {v.name: v for v in check(values)}
    assert verdicts["conflict"].passed
    assert not verdicts["straightforward"].passed


def test_one_scenario_of_noise_is_tolerated_on_an_imperfect_slice() -> None:
    values = baseline_values()
    values["conflict"] = 0.800
    # Derived from the frozen baseline rather than retyped. Hardcoding 0.941
    # against a stored 0.9412 put this two ten-thousandths the wrong side of the
    # boundary, which is the sort of thing that gets a correct gate loosened.
    base = baseline_values()["straightforward"]
    values["straightforward"] = base - (1 / 17) + 1e-9
    assert next(v for v in check(values) if v.name == "straightforward").passed


def test_a_missing_slice_is_not_a_pass() -> None:
    """A slice that vanishes from the results must fail rather than be skipped:
    silently scoring fewer scenarios is how an eval quietly stops meaning
    anything."""
    values = baseline_values()
    del values["control"]
    verdict = next(v for v in check(values) if v.name == "control")
    assert not verdict.passed
    assert "missing" in verdict.reason


def test_report_renders_every_slice() -> None:
    report = format_report(check(baseline_values()))
    for name in load_baseline()["by_slice"]:
        assert name in report
    assert "FAIL" in report  # conflict, since holding is not improving


def test_every_slice_declares_how_much_of_it_is_verified() -> None:
    """A slice with no composition would silently report 0/n and read as
    unverified when it might not be."""
    for name, base in load_baseline()["by_slice"].items():
        composition = base.get("composition")
        assert composition is not None, f"{name} declares no composition"
        assert composition["verified"] + composition["unverified"] == base["n"]


def test_conflict_is_the_only_slice_with_checked_ground_truth() -> None:
    """Pinned as a value, not asserted as a relationship (DL-10).

    If verification advances, this test fails and the baseline's composition has
    to be updated deliberately rather than drifting.
    """
    verified = {
        name: base["composition"]["verified"]
        for name, base in load_baseline()["by_slice"].items()
    }
    assert verified == {
        "adversarial": 0,
        "conflict": 7,
        "control": 0,
        "straightforward": 0,
        "superseded": 0,
    }


def test_the_report_states_the_limit_of_a_pass() -> None:
    """The gate's own caveat is printed with its results rather than living in a
    JSON field nobody reads. The must-improve slice is 7 verified of 18, and its
    pass bar is one scenario, so a pass can rest entirely on unchecked ground
    truth. That is a real bound on the claim and it belongs on the report."""
    report = format_report(check(baseline_values()))
    assert "7/18 verified" in report
    assert "0/17 verified" in report  # straightforward rests on nothing checked
    assert "not that the measurement is right" in report
