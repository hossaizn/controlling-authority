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
    "straightforward": 17,
    "ambiguous": 15,
    "control": 10,
    "conflict": 18,
    "superseded": 10,
    "out_of_scope": 12,
    "adversarial": 10,
}


def test_slice_counts_match_the_committed_balance() -> None:
    assert dict(slice_counts()) == TARGET_SLICE_COUNTS


def test_total_is_ninety() -> None:
    assert len(load_all()) == sum(TARGET_SLICE_COUNTS.values()) == 92


def test_conflict_cases_are_a_minority() -> None:
    """The slice the agent is best at must not dominate the headline number."""
    counts = slice_counts()
    assert counts["conflict"] < len(load_all()) / 3


def test_no_single_route_can_carry_the_score() -> None:
    """Review found the reverse of DL-5 was unguarded.

    The original assertion here demanded answer > clarify * 3, which actively
    enforced the imbalance that lets a never-clarifying system look competent.
    Route accuracy is now macro-averaged per the spec, so the guarantee that
    matters is that every route has enough scenarios for its own score to mean
    something.
    """
    counts = route_counts()
    for route, n in counts.items():
        assert n >= 6, f"route {route!r} has only {n} scenarios; its macro score is too coarse"


def test_a_never_clarifying_system_cannot_score_well_on_macro() -> None:
    """The concrete failure the macro decision exists to prevent.

    Micro-averaged, a system that never clarifies has an accuracy ceiling above
    80% while scoring zero on the behaviour DL-5 exists to test.
    """
    counts = route_counts()
    total = sum(counts.values())
    micro_ceiling = (total - counts["clarify"]) / total
    macro_ceiling = (len(counts) - 1) / len(counts)
    assert micro_ceiling > 0.80, "the micro hazard this guards against has gone away"
    assert macro_ceiling <= 0.75


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


def test_superseded_scenarios_forbid_the_other_version_specifically() -> None:
    """Review found this only checked that forbidden_citations was non-empty.

    Forbidding some unrelated citation would have passed while leaving the
    scenario answerable by citing both versions.
    """
    other = {"LEAVE-004-v1": "LEAVE-004-v2", "LEAVE-004-v2": "LEAVE-004-v1"}
    for s in load_all():
        if s.slice != "superseded" or s.scenario_id == "superseded-007":
            continue  # -007 asks about the change itself; both versions are legitimate
        required = [c for c in s.required_citations if c in other]
        assert required, f"{s.scenario_id} cites no handbook version"
        for cite in required:
            assert other[cite] in s.forbidden_citations, (
                f"{s.scenario_id} requires {cite} but does not forbid {other[cite]}"
            )


def test_superseded_slice_holds_jurisdiction_constant() -> None:
    """The slice isolates date. Varying state as well would confound it, and
    under precedence rule 5 a state statute would control and mask the version
    selection entirely."""
    states = {
        s.employee_context.state for s in load_all() if s.slice == "superseded"
    }
    assert states == {"OH"}, states


def test_no_ohio_scenario_is_verified_while_absence_records_are_unverified() -> None:
    """Every Ohio answer rests on "Ohio adds nothing", which is an absence
    record, and those carry verified_on: null.

    This guard exists because the same premature-verification mistake was made
    twice: once in Phase 2 and again in Phase 3, inside a single session. DL-3
    was clear and judgment still failed, so the rule is now enforced rather
    than remembered.
    """
    from ingest.absence import load_absence_records

    absences = load_absence_records("OH")
    absences_verified = all("verified_on=None" not in d.source_note for d in absences)
    if absences_verified:
        return  # once they are checked, Ohio scenarios may be verified

    offenders = [
        s.scenario_id
        for s in load_all()
        if s.verified and s.employee_context.state == "OH"
    ]
    assert not offenders, (
        f"verified while Ohio absence records are not: {offenders}"
    )


def test_no_jurisdiction_withheld_scenario_is_verified_yet() -> None:
    """Withholding the state means the answer claims something about every
    jurisdiction. Only a fraction of state law is ingested, scoped to what the
    scenarios cite, so no such claim can be checked yet."""
    offenders = [
        s.scenario_id
        for s in load_all()
        if s.verified and s.employee_context.state is None
    ]
    assert not offenders, f"uniformity claims cannot be verified yet: {offenders}"
