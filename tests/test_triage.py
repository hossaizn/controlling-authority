"""Tests for triage and for the route metric.

No network. `conftest.py` blocks the socket layer, so the caller is injected.
"""

from __future__ import annotations

from datetime import date

from agent.graph import build_graph
from agent.nodes.triage import (
    NO_FACT,
    PROMPT_VERSION,
    TRIAGE_TOOL,
    make_triage,
)
from agent.state import initial_state, nodes_visited
from domain import EmployeeContext, missing_facts
from eval.run_routes import RouteOutcome, RouteReport, format_report


class FakeCaller:
    """Records what it was asked and returns a pinned tool result."""

    def __init__(self, **result):
        self.result = {
            "route": "answer",
            "why": "answerable from the indexed policies",
            "missing_fact": NO_FACT,
            "search_query": "family care and medical leave entitlement",
            "jurisdiction": NO_FACT,
            **result,
        }
        self.system: str | None = None
        self.user: str | None = None
        self.model: str | None = None

    def call(self, system, user, tool, model, **kwargs):
        self.system, self.user, self.model = system, user, model
        return self.result


def run_triage(caller, question="can I take time off?", **ctx_kwargs):
    state = initial_state(
        question,
        employee_context=EmployeeContext(**ctx_kwargs),
        as_of=date(2026, 4, 1),
    )
    return make_triage(caller)(state), state


# --- the node ---------------------------------------------------------------


def test_the_route_and_query_reach_the_state() -> None:
    out, _ = run_triage(FakeCaller(route="answer", search_query="bereavement leave"))
    assert out["route"] == "answer"
    assert out["rewritten_query"] == "bereavement leave"


def test_a_clarify_that_names_no_fact_is_downgraded_to_answer() -> None:
    """`clarify` without a fact is unusable: the next node has nothing to ask
    about. Downgrading makes it a visible routing error instead of a crash."""
    out, _ = run_triage(FakeCaller(route="clarify", missing_fact=NO_FACT))
    assert out["route"] == "answer"
    assert out["missing_fact"] is None


def test_a_clarify_naming_a_fact_is_kept() -> None:
    out, _ = run_triage(FakeCaller(route="clarify", missing_fact="tenure_months"))
    assert out["route"] == "clarify"
    assert out["missing_fact"] == "tenure_months"


def test_supplied_context_beats_a_state_read_out_of_the_question() -> None:
    """The context comes from a system of record; the question is someone
    typing. If they disagree, the record wins."""
    out, _ = run_triage(FakeCaller(jurisdiction="NY"), state="CA")
    assert out["jurisdiction"] == "CA"


def test_a_state_named_in_the_question_is_used_when_context_has_none() -> None:
    out, _ = run_triage(FakeCaller(jurisdiction="NY"))
    assert out["jurisdiction"] == "NY"


def test_no_state_anywhere_leaves_jurisdiction_unset() -> None:
    """Unset means the retrieval filter is not applied, which is correct: an
    invented jurisdiction would filter out the law that actually governs."""
    out, _ = run_triage(FakeCaller(jurisdiction=NO_FACT))
    assert out["jurisdiction"] is None


def test_the_trace_carries_both_the_raw_and_the_rewritten_query() -> None:
    """The rewrite is the thing under suspicion, so the trace has to show what
    was replaced, not only what was sent."""
    out, _ = run_triage(
        FakeCaller(search_query="grandparent care leave"), question="can I mind my grandma?"
    )
    detail = out["trace"][0].detail
    assert detail["raw_question"] == "can I mind my grandma?"
    assert detail["query_sent_to_index"] == "grandparent care leave"


def test_the_prompt_version_is_in_the_cache_key_material() -> None:
    """Editing the prompt must invalidate cached decisions, or a run reports
    yesterday's routing for today's prompt."""
    caller = FakeCaller()
    run_triage(caller)
    assert PROMPT_VERSION in caller.system


def test_the_model_is_told_the_date_and_the_facts_it_already_has() -> None:
    caller = FakeCaller()
    run_triage(caller, tenure_months=14, state="CA")
    assert "2026-04-01" in caller.user
    assert "tenure_months: 14" in caller.user
    assert "state: CA" in caller.user


def test_facts_that_were_not_supplied_are_not_listed_as_known() -> None:
    caller = FakeCaller()
    run_triage(caller, tenure_months=14)
    assert "hours_worked_12mo" not in caller.user


def test_an_empty_context_says_so_rather_than_listing_nothing() -> None:
    caller = FakeCaller()
    run_triage(caller)
    assert "nothing supplied" in caller.user


def test_the_tool_offers_exactly_the_facts_the_schema_knows() -> None:
    """A tool enum that drifts from MissingFact would let the model return a
    fact the scenario schema cannot express, and the mismatch would surface as
    a scoring error rather than a validation one."""
    offered = set(TRIAGE_TOOL["input_schema"]["properties"]["missing_fact"]["enum"])
    assert offered == {NO_FACT, *missing_facts()}


def test_triage_drives_the_graph_down_the_route_it_chose() -> None:
    graph = build_graph(triage=make_triage(FakeCaller(route="refuse")))
    out = graph.invoke(initial_state("how do I change my 401k?"))
    assert nodes_visited(out) == ["triage", "refuse"]


# --- the metric -------------------------------------------------------------


def outcomes(*specs) -> RouteReport:
    report = RouteReport()
    for i, (expected, predicted) in enumerate(specs):
        report.outcomes.append(
            RouteOutcome(f"s-{i}", "slice", expected, predicted, None, None)
        )
    return report


def test_macro_and_micro_diverge_when_a_route_is_never_predicted() -> None:
    """The reason DL-7 changed the metric. A system that never clarifies scores
    well on micro and must not on macro."""
    report = outcomes(
        *[("answer", "answer")] * 9,
        ("clarify", "answer"),
    )
    assert report.micro_accuracy == 0.9
    assert report.macro_accuracy == 0.5


def test_a_route_with_one_scenario_counts_as_much_as_a_route_with_fifty() -> None:
    report = outcomes(*[("answer", "answer")] * 50, ("escalate", "refuse"))
    assert report.macro_accuracy == 0.5


def test_over_clarification_counts_only_scenarios_that_should_not_ask() -> None:
    report = outcomes(
        ("answer", "clarify"),
        ("answer", "answer"),
        ("refuse", "refuse"),
        ("clarify", "clarify"),
    )
    assert report.over_clarification_rate == 1 / 3


def test_under_clarification_counts_only_scenarios_that_should_ask() -> None:
    report = outcomes(
        ("clarify", "answer"),
        ("clarify", "clarify"),
        ("answer", "answer"),
    )
    assert report.under_clarification_rate == 0.5


def test_an_always_clarifying_system_is_caught_by_the_rate_not_the_accuracy() -> None:
    """Trivially safe and unusable. Macro accuracy alone would report 0.25,
    which reads as 'bad model' rather than 'wrong behaviour'."""
    report = outcomes(
        *[("answer", "clarify")] * 10,
        ("clarify", "clarify"),
    )
    assert report.over_clarification_rate == 1.0
    assert report.under_clarification_rate == 0.0


def test_missing_fact_accuracy_ignores_clarifies_that_never_fired() -> None:
    """A clarify that did not happen is an under-clarification, already counted.
    Counting it here too would punish the same failure twice and hide how good
    the questions are when they do get asked."""
    report = RouteReport(
        outcomes=[
            RouteOutcome("a", "s", "clarify", "clarify", "state", "state"),
            RouteOutcome("b", "s", "clarify", "clarify", "state", "employer_size"),
            RouteOutcome("c", "s", "clarify", "answer", "state", None),
        ]
    )
    assert report.missing_fact_accuracy == 0.5


def test_the_confusion_matrix_places_predictions_in_the_expected_row() -> None:
    matrix = outcomes(("refuse", "escalate"), ("refuse", "refuse")).confusion()
    assert matrix["refuse"]["escalate"] == 1
    assert matrix["refuse"]["refuse"] == 1
    assert matrix["escalate"]["refuse"] == 0


def test_the_report_leads_with_macro_and_labels_micro_as_weighted() -> None:
    text = format_report(outcomes(("answer", "answer"), ("clarify", "answer")))
    assert "MACRO" in text
    assert "majority-weighted" in text
