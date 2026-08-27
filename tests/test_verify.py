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


def test_number_words_are_matched_as_whole_words_not_substrings() -> None:
    """Confirmed by review: "written" contains "ten" and "none" contains "one",
    so substring matching marked 10 and 1 as supported by any corpus containing
    ordinary prose. That turned a groundedness check into a rubber stamp."""
    stated = supported_figures("the employee must give written notice; none apply")
    assert "10" not in stated
    assert "1" not in stated


def test_a_fabricated_quantity_is_caught_even_when_the_prose_looks_similar() -> None:
    hits = [hit("29 CFR 825.302", text="must give written notice of the need for leave")]
    out, _ = run(
        "You must give 10 days of notice [29 CFR 825.302].",
        hits=hits,
        res=resolution("federal", "29 CFR 825.302"),
        citations=["29 CFR 825.302"],
        caller=Exploding(),
    )
    assert out["verification"].checks["figures_appear_in_the_sources"] is False


def test_a_thousands_separator_does_not_make_a_figure_look_fabricated() -> None:
    """The corpus writes "1,250 hours" and an answer writes "1250 hours"."""
    assert "1250" in supported_figures("at least 1,250 hours of service")
    assert figures_in("you worked 1,250 hours", set()) == {"1250"}


def test_only_numbers_carrying_a_unit_count_as_quantities() -> None:
    """The earlier design excluded digits found in citations, which made 29
    exempt corpus-wide because every federal citation is "29 CFR ...". An answer
    claiming 29 workweeks passed. Requiring a unit closes it from both sides."""
    figures = figures_in(
        "Under section 825.201 you get 29 workweeks. [29 CFR 825.201]",
        {"29 CFR 825.201"},
    )
    assert figures == {"29"}


def test_the_number_twentynine_is_no_longer_exempt() -> None:
    out, _ = run(
        "You are entitled to 29 workweeks of leave. [29 CFR 825.200]",
        caller=Exploding(),
    )
    assert out["verification"].checks["figures_appear_in_the_sources"] is False


def test_any_defensible_provision_satisfies_the_controlling_citation_check() -> None:
    """On an indeterminate resolution both layers independently compel the
    outcome, so citing either is correct. Taking only the first failed an answer
    that correctly cited state because federal came first in `considered`."""
    hits = [hit("29 CFR 825.110"), hit("Cal. Gov. Code 12945.2", layer="state")]
    res = Resolution(
        controlling=None,
        rule="indeterminate",
        acceptable=["federal", "state"],
        considered=[
            LayerFinding("federal", True, "grants", "29 CFR 825.110", "", 1),
            LayerFinding("state", True, "grants", "Cal. Gov. Code 12945.2", "", 1),
        ],
    )
    out, _ = run(
        "You are eligible at 12 workweeks [Cal. Gov. Code 12945.2].",
        hits=hits,
        res=res,
        citations=["Cal. Gov. Code 12945.2"],
    )
    assert out["verification"].checks["controlling_provision_cited"] is True


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


def test_a_subsection_of_a_retrieved_provision_is_not_a_stray_citation() -> None:
    """The model writes `Cal. Gov. Code 12945.2(b)(13)` where
    `Cal. Gov. Code 12945.2` was retrieved. That is a more precise pointer into
    the same passage, and failing it wiped correct answers."""
    hits = [hit("Cal. Gov. Code 12945.2", layer="state")]
    out, _ = run(
        "You qualify [Cal. Gov. Code 12945.2(b)(13)] for 12 workweeks.",
        hits=hits,
        res=resolution("state", "Cal. Gov. Code 12945.2"),
        citations=["Cal. Gov. Code 12945.2"],
    )
    assert out["verification"].checks["citations_were_retrieved"] is True


def test_a_different_statute_sharing_a_prefix_is_still_stray() -> None:
    """`Cal. Gov. Code 12945` and `Cal. Gov. Code 12945.2` are different
    statutes. The subsection allowance must not launder one into the other, so
    the extension has to open with a bracket."""
    hits = [hit("Cal. Gov. Code 12945", layer="state")]
    out, _ = run(
        "You qualify [Cal. Gov. Code 12945.2].",
        hits=hits,
        res=resolution("state", "Cal. Gov. Code 12945"),
        citations=["Cal. Gov. Code 12945.2"],
        caller=Exploding(),
    )
    assert out["verification"].checks["citations_were_retrieved"] is False


def test_a_wholly_invented_citation_in_prose_is_caught() -> None:
    """The check this replaced could never fire: it read the citation list that
    `compose` had already filtered by the same predicate."""
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.999].",
        citations=[],
        caller=Exploding(),
    )
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


def test_an_unsupported_claim_annotates_rather_than_discards() -> None:
    """**The Step 4 change.** Entailment is one model's reading of whether a
    sentence follows from a passage, and it was silently destroying answers that
    passed every decidable check: 15 of 19 rejections had the correct
    controlling authority, 9 of them in the conflict slice.

    A referral saying "I could not confirm an answer" gives the reader nothing.
    An answer with a flagged claim gives them the provision, the reasoning, and
    a specific thing to be careful about.
    """
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.200].",
        caller=FakeCaller(supported=False, unsupported=["leave is fully paid"]),
    )
    v = out["verification"]
    assert v.passed is True, "the decidable checks all passed"
    assert v.fully_grounded is False, "but it is not fully grounded"
    assert "leave is fully paid" in v.advisories[0]
    assert v.failures == []
    assert "answer" not in out, "the answer survives"


def test_a_deterministic_failure_still_discards_the_answer() -> None:
    """The split is not severity. A citation that was not retrieved was not
    retrieved, and no reading makes shipping it acceptable."""
    out, _ = run(
        "You get 26 workweeks [29 CFR 825.200].", caller=Exploding()
    )
    assert out["verification"].passed is False
    assert REFERRAL in out["answer"]


def test_strict_mode_restores_blocking(monkeypatch) -> None:
    """A deployment weighing an unsupported claim as worse than no answer sets
    this. The posture is a product decision, so it is configurable rather than
    decided once in code."""
    monkeypatch.setenv("VERIFY_ENTAILMENT_BLOCKS", "1")
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.200].",
        caller=FakeCaller(supported=False, unsupported=["leave is fully paid"]),
    )
    assert out["verification"].passed is False
    assert out["verification"].advisories == []
    assert REFERRAL in out["answer"]


def test_an_unsupported_verdict_with_no_detail_is_still_flagged() -> None:
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.200].",
        caller=FakeCaller(supported=False, unsupported=[]),
    )
    assert out["verification"].advisories == ["a claim is not supported by its source"]
    assert out["verification"].fully_grounded is False


def test_a_clean_answer_is_fully_grounded() -> None:
    out, _ = run("You are entitled to 12 workweeks [29 CFR 825.200].")
    assert out["verification"].fully_grounded is True


def test_the_trace_says_how_many_claims_were_flagged() -> None:
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.200].",
        caller=FakeCaller(supported=False, unsupported=["a", "b"]),
    )
    assert "2 claim(s) flagged" in out["trace"][0].summary


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


def test_advisories_are_kept_apart_from_failures() -> None:
    """Two channels, not one list with severities. The eval reports a strict and
    a blocking number from them, and merging them would make that impossible."""
    out, _ = run(
        "You are entitled to 12 workweeks [29 CFR 825.200].",
        caller=FakeCaller(supported=False, unsupported=["x"]),
    )
    detail = out["trace"][0].detail
    assert detail["failures"] == []
    assert detail["advisories"]
