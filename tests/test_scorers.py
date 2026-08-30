"""Tests for the eval scorers.

**These did not exist, and that was the most serious finding of the Phase 6
review.** Every scorer could be mutated to return True unconditionally and the
whole suite stayed green: `precedence_correct`, `naive_correct`,
`addressed_what_it_should`, `verified`, `required_present`, `forbidden_present`,
and `build_baseline` swapped to use the real resolver. Both halves of the
headline comparison and all of `fully_correct` rested on arithmetic nothing
tested.

That is DL-10's finding one level up. The agent code was mutation-tested from the
start; the code that measures it was not, so a wrong number would have been
indistinguishable from a right one.
"""

from __future__ import annotations

from datetime import date

from agent.build import naive_resolve
from agent.state import LayerFinding, Resolution, VerificationResult
from eval.run_end_to_end import EndToEnd
from eval.run_precedence import PrecedenceOutcome, acceptable_set
from eval.scenarios.schema import Scenario
from retrieval.store import SearchHit


def scenario(**kw) -> Scenario:
    base = dict(
        scenario_id="s-1",
        slice="conflict",
        question="how much leave?",
        employee_context={},
        as_of_date=date(2026, 4, 1),
        expected_route="answer",
        expected_authority="state",
        required_citations=["Cal. Gov. Code 12945.2"],
    )
    base.update(kw)
    return Scenario(**base)


def resolution(controlling=None, acceptable=(), considered=()) -> Resolution:
    return Resolution(
        controlling=controlling,
        rule="statutory_floor",
        considered=list(considered),
        acceptable=list(acceptable),
    )


def hit(citation, layer="federal"):
    return SearchHit(
        chunk_id=f"c-{citation}", citation=citation, authority_layer=layer,
        jurisdiction="US", content_status="substantive", heading="h",
        text="t", score=0.5,
    )


def outcome(scen, res, routed_to="answer", naive_layer=None, must_address=()):
    return PrecedenceOutcome(
        scenario_id=scen.scenario_id, slice_name=scen.slice,
        expected=acceptable_set(scen), resolution=res, routed_to=routed_to,
        must_address=list(must_address), naive_layer=naive_layer,
    )


def end_to_end(scen, **final) -> EndToEnd:
    return EndToEnd(scen, dict(final))


# --- acceptable_set ---------------------------------------------------------


def test_a_single_expected_authority_makes_a_one_member_set() -> None:
    assert acceptable_set(scenario(expected_authority="state")) == {"state"}


def test_acceptable_authorities_makes_a_multi_member_set() -> None:
    s = scenario(expected_authority=None, acceptable_authorities=["federal", "state"])
    assert acceptable_set(s) == {"federal", "state"}


def test_a_scenario_asserting_no_authority_has_an_empty_set() -> None:
    s = scenario(expected_route="refuse", expected_authority=None, required_citations=[])
    assert acceptable_set(s) == set()


# --- precedence correctness -------------------------------------------------


def test_the_right_single_authority_is_correct() -> None:
    assert outcome(scenario(), resolution("state")).correct


def test_the_wrong_single_authority_is_not_correct() -> None:
    assert not outcome(scenario(), resolution("federal")).correct


def test_naming_two_authorities_when_one_controls_is_not_correct() -> None:
    """Subset, not intersection. A resolution calling federal defensible when
    only state controls has named an authority that does not control, and
    scoring that correct because one member matched would reward vagueness."""
    assert not outcome(scenario(), resolution(acceptable=["federal", "state"])).correct


def test_either_member_of_a_genuinely_indeterminate_pair_is_correct() -> None:
    s = scenario(expected_authority=None, acceptable_authorities=["federal", "state"])
    assert outcome(s, resolution("state")).correct
    assert outcome(s, resolution("federal")).correct


def test_resolving_nothing_is_not_correct() -> None:
    """An empty resolution must not pass by vacuous subset."""
    assert not outcome(scenario(), resolution()).correct


# --- the naive baseline -----------------------------------------------------


def test_naive_correct_compares_the_top_ranked_layer() -> None:
    assert outcome(scenario(), resolution("state"), naive_layer="state").naive_correct
    assert not outcome(scenario(), resolution("state"), naive_layer="company").naive_correct


def test_naive_resolve_takes_the_first_hit_and_nothing_else() -> None:
    """One implementation of the baseline. There were two, in `build.py` and in
    the precedence scorer, and DL-23 claimed the baseline could not drift from
    the agent while it could drift from itself."""
    state = {"retrieved": [hit("LEAVE-008", "company"), hit("29 CFR 825.200", "federal")]}
    assert naive_resolve(state)["resolution"].controlling == "company"


def test_naive_resolve_with_nothing_retrieved_controls_nothing() -> None:
    assert naive_resolve({"retrieved": []})["resolution"].controlling is None


def test_naive_resolve_fires_no_precedence_rule() -> None:
    """It is the absence of reasoning that is being measured."""
    state = {"retrieved": [hit("LEAVE-008", "company")]}
    assert naive_resolve(state)["resolution"].rule == "not_reached"


# --- must_address -----------------------------------------------------------


def test_a_named_beaten_source_counts() -> None:
    res = Resolution(
        controlling="state", rule="statutory_floor",
        non_controlling_to_address=["LEAVE-002"],
    )
    assert outcome(scenario(), res, must_address=["LEAVE-002"]).addressed_what_it_should


def test_an_unnamed_beaten_source_does_not_count() -> None:
    res = Resolution(controlling="state", rule="statutory_floor")
    assert not outcome(scenario(), res, must_address=["LEAVE-002"]).addressed_what_it_should


# --- end-to-end scoring -----------------------------------------------------


def test_verification_that_did_not_run_is_a_pass_for_a_refusal() -> None:
    """DL-25's fix, which had no regression test. Refusals assert no entitlement
    and never reach verify; scoring their absent result as a failure put
    out_of_scope at route 1.000 and fully-correct 0.000 at the same time."""
    s = scenario(expected_route="refuse", expected_authority=None, required_citations=[])
    assert end_to_end(s, route="refuse", answer="not covered").verified


def test_verification_that_did_not_run_is_not_a_pass_for_an_answer() -> None:
    """The other direction. An answering path that somehow skipped verify has
    not been checked, and must not be credited as though it had."""
    assert not end_to_end(scenario(), route="answer", answer="you get leave").verified


def test_a_failed_verification_is_not_a_pass() -> None:
    r = end_to_end(
        scenario(), route="answer", answer="x",
        verification=VerificationResult(passed=False, failures=["ungrounded"]),
    )
    assert not r.verified


def test_required_citations_are_matched_on_a_bracket_boundary() -> None:
    """DL-12's prefix trap, reintroduced in the scorer. `Cal. Gov. Code 12945`
    is a prefix of `Cal. Gov. Code 12945.2` and they are different statutes."""
    s = scenario(required_citations=["Cal. Gov. Code 12945"])
    cfra = end_to_end(s, route="answer", answer="You qualify [Cal. Gov. Code 12945.2].")
    pdl = end_to_end(s, route="answer", answer="You qualify [Cal. Gov. Code 12945].")
    assert not cfra.required_present
    assert pdl.required_present


def test_a_citation_mentioned_in_prose_counts_without_brackets() -> None:
    """The bracket-only matcher reported naming the beaten source as a flat zero.
    `compose` brackets the controlling provision and mentions the handbook it
    overrides in prose, so demanding brackets scored a correct answer wrong."""
    s = scenario(must_address=["LEAVE-002"], required_citations=["Cal. Gov. Code 12945.2"])
    r = end_to_end(
        s,
        route="answer",
        answer=(
            "You qualify [Cal. Gov. Code 12945.2]. The handbook policy LEAVE-002 "
            "requires 18 months, but it cannot fall below the statutory floor."
        ),
    )
    assert r.addressed


def test_prose_matching_still_refuses_a_prefix_of_a_longer_citation() -> None:
    s = scenario(must_address=["LEAVE-004"], required_citations=["Cal. Gov. Code 12945.2"])
    r = end_to_end(s, route="answer", answer="see policy LEAVE-004-v2 for details")
    assert not r.addressed


def test_a_forbidden_citation_in_the_answer_is_detected() -> None:
    s = scenario(required_citations=["LEAVE-004-v2"], forbidden_citations=["LEAVE-004-v1"])
    assert end_to_end(s, route="answer", answer="see [LEAVE-004-v1]").forbidden_present


def test_a_forbidden_citation_absent_from_the_answer_is_not_detected() -> None:
    s = scenario(required_citations=["LEAVE-004-v2"], forbidden_citations=["LEAVE-004-v1"])
    assert not end_to_end(s, route="answer", answer="see [LEAVE-004-v2]").forbidden_present


def test_fully_correct_requires_every_condition_at_once() -> None:
    """The headline. Each clause is checked by dropping exactly one."""
    s = scenario(required_citations=["Cal. Gov. Code 12945.2"])
    good = dict(
        route="answer",
        answer="You qualify [Cal. Gov. Code 12945.2].",
        resolution=resolution("state"),
        verification=VerificationResult(passed=True),
    )
    assert end_to_end(s, **good).fully_correct

    assert not end_to_end(s, **{**good, "route": "clarify"}).fully_correct
    assert not end_to_end(s, **{**good, "resolution": resolution("company")}).fully_correct
    assert not end_to_end(s, **{**good, "answer": "You qualify."}).fully_correct
    assert not end_to_end(
        s, **{**good, "verification": VerificationResult(passed=False)}
    ).fully_correct


def test_a_forbidden_citation_alone_makes_it_not_fully_correct() -> None:
    s = scenario(
        required_citations=["LEAVE-004-v2"], forbidden_citations=["LEAVE-004-v1"]
    )
    r = end_to_end(
        s,
        route="answer",
        answer="see [LEAVE-004-v2] and [LEAVE-004-v1]",
        resolution=Resolution(
            controlling="state", rule="statutory_floor",
            considered=[LayerFinding("state", True, "grants", "LEAVE-004-v2", "", 1)],
        ),
        verification=VerificationResult(passed=True),
    )
    assert r.required_present
    assert not r.fully_correct


# --- DL-41: run files must not collide across sampling arms ------------------


def test_an_unset_temperature_keeps_the_existing_run_filename() -> None:
    """The control arm's file is cited by DL-22 and DL-24. Adding a temperature
    suffix unconditionally would rename it and break both references while the
    run still passed."""
    from eval.run_triage import temperature_slug

    assert temperature_slug(None) == ""


def test_each_temperature_gets_its_own_run_file() -> None:
    """Two arms writing one path is a paired comparison with one arm in it.

    The second run would overwrite the first and the delta would read as zero,
    which is DL-41's own registered prediction arriving by accident.
    """
    from eval.run_triage import temperature_slug

    slugs = [temperature_slug(t) for t in (None, 0.0, 0.7, 1.0)]
    assert len(set(slugs)) == 4
    assert all("/" not in s and "." not in s for s in slugs)
