"""Tests for the precedence rules.

The cases are the real ones. Where a test reproduces a scenario it names it, so
that a change to the rules shows up against the ground truth it has to satisfy
rather than against invented examples that agree with whatever the code does.
"""

from __future__ import annotations

import pytest

from agent.precedence import PrecedenceError, resolve_precedence
from agent.state import LayerFinding


def finding(layer, outcome="silent", rank=None, citation=None, says=""):
    return LayerFinding(
        layer=layer,
        speaks_to_question=outcome != "silent",
        outcome=outcome,
        citation=citation,
        says=says,
        generosity_rank=rank,
    )


# --- rule 4: silence is not permission --------------------------------------


def test_a_silent_layer_does_not_override_a_speaking_one() -> None:
    """`conflict-008`, Ohio. No Ohio provision restricts forfeiture, so the
    handbook clause stands and the forfeited days are gone."""
    r = resolve_precedence([
        finding("federal"),
        finding("state"),
        finding("company", "grants", 1, "LEAVE-008"),
    ])
    assert r.controlling == "company"
    assert r.rule == "silence_is_not_permission"


def test_every_layer_silent_resolves_to_nothing_controlling() -> None:
    """Not an error and not a guess. Nothing in the corpus bears on it, which
    is a refusal, and the route for that was decided before this ran."""
    r = resolve_precedence([finding("federal"), finding("state"), finding("company")])
    assert r.controlling is None
    assert r.rule == "silence_is_not_permission"


def test_a_layer_that_speaks_and_denies_still_counts_as_speaking() -> None:
    """`conflict-005`, the single most valuable case in the set. Federal FMLA
    does not cover an ordinary sick grandparent, Ohio adds nothing, the handbook
    is silent. The answer is "no", which is an answer rather than a refusal, and
    federal is what determined it."""
    r = resolve_precedence([
        finding("federal", "denies", 1, "29 CFR 825.122"),
        finding("state"),
        finding("company"),
    ])
    assert r.controlling == "federal"


# --- rule 2: policy may exceed, never reduce --------------------------------


def test_a_handbook_above_the_floor_controls() -> None:
    """`conflict-002`. The handbook grants 10 paid bereavement days against a
    5-day statutory floor. An agent that learned "statute always wins" answers
    with the minimum and is wrong."""
    r = resolve_precedence([
        finding("state", "grants", 2, "Cal. Gov. Code 12945.7"),
        finding("company", "grants", 1, "LEAVE-003"),
    ])
    assert r.controlling == "company"
    assert r.rule == "policy_may_exceed"


def test_a_handbook_below_the_floor_is_unenforceable() -> None:
    """`conflict-001`'s mechanism. The handbook demands 18 months; CFRA sets it
    at 12. The handbook is the closest semantic match and the wrong answer."""
    r = resolve_precedence([
        finding("state", "grants", 1, "Cal. Gov. Code 12945.2"),
        finding("company", "denies", 2, "LEAVE-002"),
    ])
    assert r.controlling == "state"
    assert r.rule == "statutory_floor"


def test_the_beaten_handbook_is_named_for_the_reader() -> None:
    """Someone who already read the handbook needs to know why the answer
    differs from it, or the answer is useless to them."""
    r = resolve_precedence([
        finding("state", "grants", 1, "Cal. Gov. Code 12945.2"),
        finding("company", "denies", 2, "LEAVE-002"),
    ])
    assert "LEAVE-002" in r.non_controlling_to_address


# --- rule 1: the employee-favourable statutory floor ------------------------


def test_the_more_generous_statute_governs_not_the_state_one() -> None:
    """`conflict-017`. At exactly twelve months federal says "at least 12" and
    California says "more than 12", so federal is met and the state test is not.
    The rule is not "state beats federal"."""
    r = resolve_precedence([
        finding("federal", "grants", 1, "29 CFR 825.110"),
        finding("state", "denies", 2, "Cal. Gov. Code 12945.2"),
    ])
    assert r.controlling == "federal"
    assert r.rule == "statutory_floor"


def test_state_governs_where_it_exceeds_federal() -> None:
    """`conflict-010`. A 20-person employer is under the federal 50 threshold
    and over California's five."""
    r = resolve_precedence([
        finding("federal", "denies", 2, "29 CFR 825.104"),
        finding("state", "grants", 1, "Cal. Gov. Code 12945.2"),
    ])
    assert r.controlling == "state"


# --- rule 5: concurrence, the one a review found missing --------------------


def test_a_handbook_restating_a_statute_concurs_rather_than_controls() -> None:
    """Rule 5, added after DL-7 found precedence unscoreable without it. Strike
    the handbook and the entitlement survives, so the statute is what compels."""
    r = resolve_precedence([
        finding("state", "grants", 1, "Cal. Lab. Code 246"),
        finding("company", "grants", 1, "LEAVE-004-v2"),
    ])
    assert r.controlling == "state"
    assert r.rule == "concurrence_tie_break"


def test_federal_and_state_tying_is_indeterminate_rather_than_arbitrary() -> None:
    """`conflict-001`. At 14 months with 1,800 hours at a 400-person employer she
    satisfies both tests. Rule 5 orders statute above policy without ordering
    federal against state, so demanding one would score a defensible answer
    wrong."""
    r = resolve_precedence([
        finding("federal", "grants", 1, "29 CFR 825.110"),
        finding("state", "grants", 1, "Cal. Gov. Code 12945.2"),
        finding("company", "denies", 2, "LEAVE-002"),
    ])
    assert r.controlling is None
    assert r.rule == "indeterminate"
    assert sorted(r.acceptable) == ["federal", "state"]
    assert sorted(r.defensible) == ["federal", "state"]


def test_a_three_way_tie_still_drops_the_concurring_handbook() -> None:
    """The handbook does not become controlling by agreeing with two statutes
    instead of one."""
    r = resolve_precedence([
        finding("federal", "grants", 1, "29 CFR 825.200"),
        finding("state", "grants", 1, "Cal. Gov. Code 12945.2"),
        finding("company", "grants", 1, "LEAVE-001"),
    ])
    assert sorted(r.acceptable) == ["federal", "state"]
    assert "company" not in r.defensible


# --- integrity --------------------------------------------------------------


def test_a_denying_layer_ranked_above_a_granting_one_raises() -> None:
    """The one way a model-supplied ranking can be checked against itself.

    Left unchecked, a handbook that grants leave would lose to a statute that
    merely does not require it, and the employee would be told no on the
    strength of a rule that was only ever a floor.
    """
    with pytest.raises(PrecedenceError, match="denies is ranked"):
        resolve_precedence([
            finding("federal", "denies", 1, "29 CFR 825.122"),
            finding("company", "grants", 2, "LEAVE-001"),
        ])


def test_a_speaking_layer_with_no_rank_raises() -> None:
    with pytest.raises(PrecedenceError, match="no generosity_rank"):
        resolve_precedence([finding("company", "grants", None, "LEAVE-001")])


def test_duplicate_layers_raise() -> None:
    with pytest.raises(PrecedenceError, match="one finding per layer"):
        resolve_precedence([
            finding("company", "grants", 1, "LEAVE-001"),
            finding("company", "grants", 1, "LEAVE-002"),
        ])


def test_every_layer_is_recorded_including_the_silent_ones() -> None:
    """The trace has to show what was considered and rejected, not only what
    won. A resolution that lists one layer is indistinguishable from one that
    never looked at the others."""
    r = resolve_precedence([
        finding("federal"),
        finding("state", "grants", 1, "Cal. Gov. Code 12945.2"),
        finding("company", "denies", 2, "LEAVE-002"),
    ])
    assert {f.layer for f in r.considered} == {"federal", "state", "company"}


def test_the_same_findings_always_produce_the_same_resolution() -> None:
    """The reason this is code. A model asked the same question twice may not
    answer it the same way, and precedence is not a judgment call."""
    findings = [
        finding("federal", "grants", 2, "29 CFR 825.200"),
        finding("state", "grants", 1, "Cal. Gov. Code 12945.2"),
        finding("company", "denies", 3, "LEAVE-002"),
    ]
    first = resolve_precedence(findings)
    for _ in range(5):
        assert resolve_precedence(findings) == first
