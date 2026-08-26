"""Tests for the compose node."""

from __future__ import annotations

from datetime import date

from agent.nodes.compose import DISCLAIMER, PROMPT_VERSION, _resolution_block, make_compose
from agent.state import LayerFinding, Resolution, initial_state
from retrieval.store import SearchHit


class FakeCaller:
    def __init__(self, answer="You are entitled to 12 workweeks [29 CFR 825.200].",
                 citations=("29 CFR 825.200",)):
        self.result = {"answer": answer, "citations": list(citations)}
        self.system = self.user = None

    def call(self, system, user, tool, model, **kw):
        self.system, self.user = system, user
        return self.result


def hit(citation, layer="federal", text="body", heading="h"):
    return SearchHit(
        chunk_id=f"c-{citation}", citation=citation, authority_layer=layer,
        jurisdiction="US", content_status="substantive", heading=heading,
        text=text, score=0.5,
    )


def finding(layer, outcome="silent", rank=None, citation=None, says=""):
    return LayerFinding(layer, outcome != "silent", outcome, citation, says, rank)


def run(resolution, hits=None, caller=None, question="how much leave do I get?"):
    state = initial_state(question, as_of=date(2026, 4, 1))
    state["retrieved"] = hits if hits is not None else [hit("29 CFR 825.200")]
    state["resolution"] = resolution
    caller = caller or FakeCaller()
    return make_compose(caller)(state), caller


CONTROLLING = Resolution(
    controlling="federal",
    rule="statutory_floor",
    considered=[finding("federal", "grants", 1, "29 CFR 825.200", "12 workweeks")],
)


# --- the answer -------------------------------------------------------------


def test_the_disclaimer_is_appended_rather_than_left_to_the_model() -> None:
    """A disclaimer the model can forget is a disclaimer that will be forgotten."""
    out, _ = run(CONTROLLING)
    assert out["answer"].endswith(DISCLAIMER)


def test_the_model_is_told_not_to_write_its_own_disclaimer() -> None:
    _, caller = run(CONTROLLING)
    assert "One is appended for you" in caller.system


def test_citations_reach_the_state() -> None:
    out, _ = run(CONTROLLING)
    assert out["citations"] == ["29 CFR 825.200"]


def test_a_citation_that_was_never_retrieved_is_stripped() -> None:
    """A model cannot police its own citing, and an answer resting on a provision
    nobody produced is worse than no answer."""
    caller = FakeCaller(citations=["29 CFR 825.200", "29 CFR 825.999"])
    out, _ = run(CONTROLLING, caller=caller)
    assert out["citations"] == ["29 CFR 825.200"]
    assert out["trace"][0].detail["citations_not_retrieved"] == ["29 CFR 825.999"]


# --- the controlling authority is an input, not a decision ------------------


def test_the_prompt_states_which_layer_controls() -> None:
    _, caller = run(CONTROLLING)
    assert "federal law controls" in caller.user


def test_the_model_is_told_it_is_not_re_deciding_precedence() -> None:
    """Composing first and checking afterwards would mean arguing a model out of
    an answer it had already committed to."""
    _, caller = run(CONTROLLING)
    assert "ALREADY been decided" in caller.system


def test_an_indeterminate_resolution_says_either_may_be_cited() -> None:
    """Demanding one authority where two independently compel the same outcome
    would force a defensible answer into a wrong shape."""
    resolution = Resolution(
        controlling=None, rule="indeterminate",
        acceptable=["federal", "state"],
        considered=[finding("federal", "grants", 1, "29 CFR 825.110", "12 months")],
    )
    _, caller = run(resolution)
    assert "each independently compel" in caller.user


# --- the beaten source ------------------------------------------------------


def test_sources_to_address_are_named_as_a_requirement() -> None:
    """The requirement most likely to be skipped, and the one that makes the
    answer usable to someone holding the handbook."""
    resolution = Resolution(
        controlling="state", rule="statutory_floor",
        considered=[finding("state", "grants", 1, "Cal. Gov. Code 12945.2", "12 months")],
        non_controlling_to_address=["LEAVE-002"],
    )
    _, caller = run(resolution)
    assert "MUST address" in caller.user
    assert "LEAVE-002" in caller.user


def test_no_sources_to_address_leaves_the_requirement_out() -> None:
    _, caller = run(CONTROLLING)
    assert "MUST address" not in caller.user


def test_every_layer_appears_in_the_block_including_silent_ones() -> None:
    """The model needs to know a layer was checked and said nothing, which is
    different from it not having been checked."""
    block = _resolution_block(Resolution(
        controlling="company", rule="silence_is_not_permission",
        considered=[
            finding("federal"),
            finding("state"),
            finding("company", "grants", 1, "LEAVE-008", "10 days"),
        ],
    ))
    assert "federal: silent" in block
    assert "state: silent" in block
    assert "[LEAVE-008]" in block


# --- degrading rather than inventing ----------------------------------------


def test_no_controlling_authority_produces_a_referral_without_calling_a_model() -> None:
    """Asked to write an answer with no authority behind it, a model will do it,
    fluently."""

    class Exploding:
        def call(self, *a, **k):
            raise AssertionError("a model was called with nothing controlling")

    out, _ = run(
        Resolution(controlling=None, rule="silence_is_not_permission"),
        caller=Exploding(),
    )
    assert out["citations"] == []
    assert "ask your HR team" in out["answer"]
    assert out["answer"].endswith(DISCLAIMER)


def test_a_missing_resolution_degrades_rather_than_raising() -> None:
    state = initial_state("q", as_of=date(2026, 4, 1))
    state["retrieved"] = [hit("29 CFR 825.200")]

    class Exploding:
        def call(self, *a, **k):
            raise AssertionError("a model was called with no resolution")

    out = make_compose(Exploding())(state)
    assert out["citations"] == []


def test_the_prompt_version_is_in_the_cache_key_material() -> None:
    _, caller = run(CONTROLLING)
    assert PROMPT_VERSION in caller.system
