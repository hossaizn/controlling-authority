"""Tests for the verify node.

Four of the five checks are deterministic, so most of these need no model at
all. That is the point of the design: code cannot share a blind spot with the
model that wrote the answer.
"""

from __future__ import annotations

from datetime import date

from agent.nodes.verify import (
    REFERRAL,
    citations_in,
    figures_in,
    make_verify,
    supported_figures,
)
from agent.state import LayerFinding, Resolution, initial_state
from retrieval.store import SearchHit


class FakeCaller:
    def __init__(self, supported=True, unsupported=()):
        self.result = {
            "every_claim_supported": supported,
            "unsupported": list(unsupported),
        }
        self.calls = 0

    def call(self, system, user, tool, model, **kw):
        self.calls += 1
        return self.result


class Exploding:
    def call(self, *a, **k):
        raise AssertionError("a model was called when it should not have been")


def hit(citation, text="twelve workweeks, being 12 workweeks of leave", layer="federal"):
    return SearchHit(
        chunk_id=f"c-{citation}", citation=citation, authority_layer=layer,
        jurisdiction="US", content_status="substantive", heading="h",
        text=text, score=0.5,
    )


def resolution(layer="federal", citation="29 CFR 825.200"):
    return Resolution(
        controlling=layer,
        rule="statutory_floor",
        considered=[LayerFinding(layer, True, "grants", citation, "12 workweeks", 1)],
    )


def run(answer, hits=None, res=None, citations=None, caller=None):
    state = initial_state("how much leave?", as_of=date(2026, 4, 1))
    state["retrieved"] = hits if hits is not None else [hit("29 CFR 825.200")]
    state["resolution"] = res if res is not None else resolution()
    state["answer"] = answer
    state["citations"] = citations if citations is not None else ["29 CFR 825.200"]
    caller = caller or FakeCaller()
    return make_verify(caller)(state), caller


# --- helpers ----------------------------------------------------------------


def test_citations_are_matched_against_what_was_retrieved() -> None:
    """Not parsed out of the prose: a regex for things that look like citations
    would miss handbook ids and invent matches from ordinary numbers."""
    found = citations_in("see [LEAVE-008] and [29 CFR 825.200]", {"LEAVE-008", "X"})
    assert found == {"LEAVE-008"}


def test_figures_exclude_the_numbers_inside_citations() -> None:
    """Without this, `29 CFR 825.200` contributes 29 and 825.200 as though the
    answer had quoted them as quantities, and every answer looks unsupported."""
    figures = figures_in("You get 12 weeks [29 CFR 825.200].", {"29 CFR 825.200"})
    assert figures == {"12"}


def test_a_bare_section_number_is_not_treated_as_a_quantity() -> None:
    """Found by this check failing valid answers. Stripping the full citation is
    not enough: an answer that cites [29 CFR 825.201] and then says "section
    825.201" leaves a bare number behind."""
    figures = figures_in(
        "Under section 825.201 you get 12 weeks [29 CFR 825.201].",
        {"29 CFR 825.201"},
    )
    assert figures == {"12"}


def test_a_spelled_out_number_in_the_sources_supports_a_digit_in_the_answer() -> None:
    """Legal drafting spells numbers out far more often than it uses digits, and
    the corpus is legal drafting. A literal digit search reported correct figures
    as unsupported."""
    stated = supported_figures("eligible for twenty-six weeks of benefits")
    assert "26" in stated


def test_common_number_words_map_to_their_digits() -> None:
    stated = supported_figures("twelve workweeks, fifty employees, five days")
    assert {"12", "50", "5"} <= stated


def test_a_figure_stated_only_in_words_still_passes_the_check() -> None:
    hits = [hit("N.Y. WCL 203", text="after twenty-six weeks of employment")]
    out, _ = run(
        "You qualify after 26 weeks [N.Y. WCL 203].",
        hits=hits,
        res=resolution("state", "N.Y. WCL 203"),
        citations=["N.Y. WCL 203"],
    )
    assert out["verification"].checks["figures_appear_in_the_sources"] is True


# --- deterministic checks ---------------------------------------------------


def test_a_grounded_answer_passes() -> None:
    out, _ = run("You are entitled to 12 workweeks [29 CFR 825.200].")
    assert out["verification"].passed
    assert "answer" not in out  # untouched


def test_an_answer_citing_something_never_retrieved_fails() -> None:
    out, caller = run(
        "You get 12 workweeks [29 CFR 825.999].",
        citations=["29 CFR 825.999"],
        caller=Exploding(),
    )
    assert not out["verification"].passed
    assert out["verification"].checks["citations_were_retrieved"] is False


def test_an_entitlement_asserted_with_no_citation_fails() -> None:
    out, _ = run("You are entitled to twelve workweeks.", citations=[], caller=Exploding())
    assert not out["verification"].passed
    assert "without citing" in out["verification"].failures[0]


def test_an_answer_that_cites_the_wrong_layer_fails() -> None:
    """Reaching the right outcome from a provision the precedence rules did not
    select is right by luck, not correctness."""
    hits = [hit("29 CFR 825.200"), hit("LEAVE-008", layer="company")]
    out, _ = run(
        "You get 10 days [LEAVE-008].",
        hits=hits,
        res=resolution("federal", "29 CFR 825.200"),
        citations=["LEAVE-008"],
        caller=Exploding(),
    )
    assert not out["verification"].passed
    assert "controlling provision" in out["verification"].failures[0]


def test_a_figure_not_in_the_sources_fails() -> None:
    out, _ = run(
        "You are entitled to 26 workweeks [29 CFR 825.200].", caller=Exploding()
    )
    assert not out["verification"].passed
    assert out["verification"].checks["figures_appear_in_the_sources"] is False


def test_a_figure_that_is_in_the_sources_passes() -> None:
    out, _ = run("You are entitled to 12 workweeks [29 CFR 825.200].")
    assert out["verification"].checks["figures_appear_in_the_sources"] is True


# --- the model check is last, not the gate ----------------------------------


def test_the_model_is_not_called_when_a_deterministic_check_already_failed() -> None:
    """No point paying to grade an answer that is being discarded."""
    caller = FakeCaller()
    out, caller = run(
        "You get 26 workweeks [29 CFR 825.200].", caller=caller
    )
    assert caller.calls == 0
    assert not out["verification"].passed


def test_an_unsupported_claim_fails_verification() -> None:
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.200].",
        caller=FakeCaller(supported=False, unsupported=["leave is fully paid"]),
    )
    assert not out["verification"].passed
    assert "leave is fully paid" in out["verification"].failures[0]


def test_an_unsupported_verdict_with_no_detail_still_fails() -> None:
    """`list.extend` returns None, so an `extend(...) or append(...)` idiom
    appends unconditionally. This pins the branch that bug lived in."""
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.200].",
        caller=FakeCaller(supported=False, unsupported=[]),
    )
    assert not out["verification"].passed
    assert out["verification"].failures == ["a claim is not supported by its source"]


def test_a_supported_answer_records_the_entailment_check_as_having_run() -> None:
    out, caller = run("You are entitled to 12 workweeks [29 CFR 825.200].")
    assert caller.calls == 1
    assert out["trace"][0].detail["entailment_checked"] is True


# --- degrading --------------------------------------------------------------


def test_a_failed_answer_is_replaced_with_a_referral() -> None:
    """Shipping an ungrounded answer is worse than shipping none, because the
    reader cannot tell the difference."""
    out, _ = run("You get 26 workweeks [29 CFR 825.200].", caller=Exploding())
    assert REFERRAL in out["answer"]
    assert out["citations"] == []


def test_a_referral_from_compose_is_not_failed_for_having_no_citation() -> None:
    """Nothing controlled, so there was no entitlement to cite. Failing it would
    turn a correct refusal into a verification error."""
    out, _ = run(
        "I could not find a policy that covers this.",
        res=Resolution(controlling=None, rule="silence_is_not_permission"),
        citations=[],
        caller=Exploding(),
    )
    assert out["verification"].passed
