"""Tests for the clarify and retrieve nodes."""

from __future__ import annotations

from datetime import date

import pytest

from agent.nodes.clarify import QUESTIONS, clarify
from agent.nodes.retrieve import make_retrieve
from agent.state import AgentState, TraceEvent, initial_state
from domain import EmployeeContext, missing_facts
from retrieval.store import SearchHit


def state_with(fact, trace=()):
    s = initial_state("can I take time off?", as_of=date(2026, 4, 1))
    s["missing_fact"] = fact
    s["trace"] = list(trace)
    return s


# --- clarify ----------------------------------------------------------------


def test_every_missing_fact_has_a_question() -> None:
    """A fact with no question emits an empty clarification, which reads as a
    system failure to the person who asked."""
    assert set(QUESTIONS) == set(missing_facts())


def test_the_question_names_the_fact_that_was_missing() -> None:
    out = clarify(state_with("tenure_months"))
    assert out["trace"][0].detail["missing_fact"] == "tenure_months"
    assert QUESTIONS["tenure_months"] in out["answer"]


def test_the_reason_from_triage_is_carried_into_the_question() -> None:
    """Triage already wrote a sentence for the asker explaining why the fact
    matters. Reusing it costs nothing and keeps the explanation consistent with
    the decision that produced it."""
    trace = [TraceEvent(node="triage", summary="personal leave depends on tenure")]
    out = clarify(state_with("tenure_months", trace))
    assert "personal leave depends on tenure" in out["answer"]


def test_a_clarification_with_no_triage_reason_is_still_a_usable_question() -> None:
    out = clarify(state_with("state"))
    assert out["answer"] == QUESTIONS["state"]


def test_clarify_never_asserts_an_entitlement() -> None:
    """A clarify carries no citations: it makes no claim about what anyone is
    owed, so there is nothing to cite."""
    assert clarify(state_with("employer_size"))["citations"] == []


def test_clarify_with_no_fact_raises_rather_than_asking_vaguely() -> None:
    """A question that names nothing cannot be answered and wastes the turn."""
    with pytest.raises(ValueError, match="nothing to ask about"):
        clarify(state_with(None))


# --- retrieve ---------------------------------------------------------------


class FakeStore:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.calls: list[dict] = []

    def search(self, query, jurisdiction=None, as_of=None, limit=10, **kw):
        self.calls.append(
            {"query": query, "jurisdiction": jurisdiction, "as_of": as_of, "limit": limit}
        )
        return self.hits[:limit]


def hit(citation, layer="federal", status="substantive"):
    return SearchHit(
        chunk_id=f"c-{citation}",
        citation=citation,
        authority_layer=layer,
        jurisdiction="US",
        content_status=status,
        heading="h",
        text="t",
        score=0.5,
    )


def run_retrieve(store, **overrides) -> AgentState:
    s = initial_state(
        "can I mind my grandma?",
        employee_context=EmployeeContext(),
        as_of=date(2026, 4, 1),
    )
    s.update(overrides)
    return make_retrieve(store)(s)


def test_the_rewritten_query_is_what_reaches_the_index() -> None:
    """The substitution the frozen baseline exists to watch."""
    store = FakeStore()
    run_retrieve(store, rewritten_query="grandparent care leave")
    assert store.calls[0]["query"] == "grandparent care leave"


def test_a_missing_rewrite_falls_back_to_the_raw_question() -> None:
    store = FakeStore()
    run_retrieve(store, rewritten_query=None)
    assert store.calls[0]["query"] == "can I mind my grandma?"


def test_the_jurisdiction_filter_is_passed_through() -> None:
    store = FakeStore()
    run_retrieve(store, jurisdiction="CA", rewritten_query="q")
    assert store.calls[0]["jurisdiction"] == "CA"


def test_no_jurisdiction_searches_everywhere_rather_than_guessing() -> None:
    """An invented jurisdiction filters out the law that actually governs, and
    the failure is silent because the result still looks like a result."""
    store = FakeStore()
    run_retrieve(store, jurisdiction=None, rewritten_query="q")
    assert store.calls[0]["jurisdiction"] is None


def test_the_as_of_date_is_passed_through_unchanged() -> None:
    store = FakeStore()
    run_retrieve(store, rewritten_query="q")
    assert store.calls[0]["as_of"] == date(2026, 4, 1)


def test_hits_reach_the_state() -> None:
    store = FakeStore([hit("29 CFR 825.200"), hit("LEAVE-001", "company")])
    out = run_retrieve(store, rewritten_query="q")
    assert [h.citation for h in out["retrieved"]] == ["29 CFR 825.200", "LEAVE-001"]


def test_the_trace_records_which_layers_came_back() -> None:
    """Which authority layers are present is what `resolve` reasons over, so it
    belongs in the visible trace rather than only in the hit list."""
    store = FakeStore([hit("29 CFR 825.200"), hit("LEAVE-001", "company")])
    detail = run_retrieve(store, rewritten_query="q")["trace"][0].detail
    assert detail["layers_found"] == ["company", "federal"]


def test_absence_records_are_called_out_separately() -> None:
    """A retrieval miss and a recorded silence demand opposite responses and
    must never look alike downstream."""
    store = FakeStore([hit("OH-absent-parental", "state", "absent"), hit("LEAVE-002")])
    detail = run_retrieve(store, rewritten_query="q")["trace"][0].detail
    assert detail["absence_records"] == ["OH-absent-parental"]


def test_an_empty_result_is_recorded_rather_than_swallowed() -> None:
    out = run_retrieve(FakeStore([]), rewritten_query="q")
    assert out["retrieved"] == []
    assert "0 passages" in out["trace"][0].summary


def test_a_handbook_passage_is_topped_up_when_none_ranked_high_enough() -> None:
    """The employee has probably already read the handbook, and the answer has to
    reconcile itself with what they read. Three `must_address` scenarios failed
    for the plain reason that no company passage was in the top ten."""
    store = FakeStore([hit(f"29 CFR 825.{i}") for i in range(10)] + [hit("LEAVE-001", "company")])
    out = run_retrieve(store, rewritten_query="q")
    assert "LEAVE-001" in [h.citation for h in out["retrieved"]]


def test_the_top_ranked_passages_are_not_displaced_by_the_top_up() -> None:
    """Appended, not substituted. Displacing a statutory passage to make room
    would change the ranking the regression gate measures."""
    ranked = [hit(f"29 CFR 825.{i}") for i in range(10)]
    store = FakeStore(ranked + [hit("LEAVE-001", "company")])
    out = run_retrieve(store, rewritten_query="q")
    assert [h.citation for h in out["retrieved"]][:10] == [h.citation for h in ranked]


def test_no_top_up_happens_when_the_handbook_already_ranked() -> None:
    store = FakeStore([hit("LEAVE-008", "company")] + [hit(f"29 CFR 825.{i}") for i in range(9)])
    out = run_retrieve(store, rewritten_query="q")
    assert sum(h.authority_layer == "company" for h in out["retrieved"]) == 1


# --- stage 3: guaranteed presence -------------------------------------------


def test_the_recorded_silence_guarantee_is_not_active() -> None:
    """Built and measured in DL-28, then not adopted: it delivered zero against a
    pre-registered bar of two scenarios. This pins that it stays out, so adding
    it back is a deliberate act with a number attached rather than a drift."""
    from agent.nodes.retrieve import GUARANTEES, UNADOPTED_RECORDED_SILENCE

    assert [g.name for g in GUARANTEES] == ["handbook"]
    assert UNADOPTED_RECORDED_SILENCE not in GUARANTEES


def test_an_unadopted_guarantee_still_works_when_asked_for_explicitly() -> None:
    """Carried rather than deleted, because the calculus changes with corpus
    size. It has to keep working or re-adopting it means rewriting it."""
    from agent.nodes.retrieve import UNADOPTED_RECORDED_SILENCE, apply_guarantees

    ranked = [hit(f"29 CFR 825.{i}") for i in range(10)]
    deeper = ranked + [hit("OH-absent-parental", "state", "absent")]
    out, fired = apply_guarantees(ranked, deeper, (UNADOPTED_RECORDED_SILENCE,))
    assert fired == ["recorded_silence"]
    assert "OH-absent-parental" in [h.citation for h in out]


def test_a_guarantee_with_nothing_to_add_is_not_an_error() -> None:
    """Some questions have no handbook policy and some jurisdictions have no
    absence record on the topic. Neither is a failure."""
    store = FakeStore([hit(f"29 CFR 825.{i}") for i in range(10)])
    out = run_retrieve(store, rewritten_query="q")
    assert out["trace"][0].detail["guarantees_fired"] == []
    assert len(out["retrieved"]) == 10


def test_a_guarantee_never_displaces_a_ranked_passage() -> None:
    """Appended, never substituted, so the ranking the frozen baseline measures
    is untouched and a guarantee can only add evidence."""
    ranked = [hit(f"29 CFR 825.{i}") for i in range(10)]
    store = FakeStore(ranked + [hit("LEAVE-001", "company")])
    out = run_retrieve(store, rewritten_query="q")
    assert [h.citation for h in out["retrieved"]][:10] == [h.citation for h in ranked]


def test_a_guarantee_does_not_duplicate_a_passage_another_already_added() -> None:
    """An absence record in the company layer would satisfy both predicates."""
    from agent.nodes.retrieve import apply_guarantees

    ranked = [hit(f"29 CFR 825.{i}") for i in range(10)]
    both = hit("LEAVE-000", "company", "absent")
    out, fired = apply_guarantees(ranked, ranked + [both])
    assert [h.chunk_id for h in out].count(both.chunk_id) == 1
