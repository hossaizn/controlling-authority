"""Tests for the graph skeleton and its state.

Every assertion here was checked by mutation: the implementation was broken in
the specific way the test claims to catch, and the test was confirmed to fail.
DL-10 is the reason. Tests asserting relationships rather than values passed
against three separate wrong implementations of the date arithmetic.
"""

from __future__ import annotations

from datetime import date

import pytest

from agent.graph import build_graph
from agent.state import (
    AgentState,
    TraceEvent,
    initial_state,
    nodes_visited,
)
from domain import EmployeeContext, missing_facts

ANSWER_PATH = ["triage", "retrieve", "resolve", "compose", "verify"]


def run(graph, **kwargs) -> AgentState:
    return graph.invoke(initial_state(kwargs.pop("question", "how much leave?"), **kwargs))


def triage_choosing(route: str):
    def node(state: AgentState) -> dict:
        return {"trace": [TraceEvent(node="triage", summary=route)], "route": route}

    return node


def test_the_default_graph_reaches_a_terminal_state() -> None:
    assert nodes_visited(run(build_graph())) == ANSWER_PATH


def test_the_trace_records_every_node_the_run_passed_through() -> None:
    """Not just the last one, and in order. The trace is the demo's product."""
    trace = run(build_graph())["trace"]
    assert [e.node for e in trace] == ANSWER_PATH
    assert all(isinstance(e, TraceEvent) for e in trace)


def test_a_node_cannot_shorten_the_trace() -> None:
    """The append-only guarantee, which is structural rather than a convention.

    `greedy` returns a single-element trace. Without the reducer that would
    replace everything before it and the run would end with one event. With it,
    the value is added.
    """

    def greedy(state: AgentState) -> dict:
        return {"trace": [TraceEvent(node="resolve", summary="greedy")]}

    trace = run(build_graph(resolve=greedy))["trace"]
    assert [e.node for e in trace] == ANSWER_PATH
    assert trace[0].node == "triage"


def test_a_node_returning_no_trace_does_not_erase_the_trace() -> None:
    def silent(state: AgentState) -> dict:
        return {}

    assert nodes_visited(run(build_graph(compose=silent))) == [
        "triage",
        "retrieve",
        "resolve",
        "verify",
    ]


@pytest.mark.parametrize("route", ["clarify", "refuse", "escalate"])
def test_terminal_routes_never_reach_the_answering_path(route: str) -> None:
    """A refusal that ran `compose` would have drafted an answer it then threw
    away, at cost, and the trace would show it reasoning about entitlement it had
    already decided not to assert."""
    visited = nodes_visited(run(build_graph(triage=triage_choosing(route))))
    assert visited == ["triage", route]
    for skipped in ("retrieve", "resolve", "compose", "verify"):
        assert skipped not in visited


def test_a_triage_that_sets_no_route_raises_rather_than_answering() -> None:
    """Falling through to `retrieve` would answer a question the system had
    already failed to classify, indistinguishably from a normal run."""

    def routeless(state: AgentState) -> dict:
        return {"trace": [TraceEvent(node="triage", summary="undecided")]}

    with pytest.raises(ValueError, match="refusing to guess"):
        run(build_graph(triage=routeless))


def test_an_unknown_route_raises() -> None:
    with pytest.raises(ValueError, match="unknown route"):
        run(build_graph(triage=triage_choosing("maybe")))


def test_as_of_is_carried_through_untouched() -> None:
    """A date silently replaced with today answers a 2023 question with 2026 law
    and reports no error (DL-16)."""
    asked = date(2023, 3, 1)
    assert run(build_graph(), as_of=asked)["as_of"] == asked


def test_initial_state_defaults_the_date_once_at_the_edge() -> None:
    assert initial_state("q")["as_of"] == date.today()


def test_initial_state_supplies_an_empty_context_rather_than_none() -> None:
    """`None` context and a context with nothing in it are different bugs; only
    one of them is representable if the default is an empty model."""
    assert initial_state("q")["employee_context"] == EmployeeContext()


def test_employee_context_is_preserved_for_later_nodes() -> None:
    ctx = EmployeeContext(state="CA", tenure_months=14)
    assert run(build_graph(), employee_context=ctx)["employee_context"] == ctx


def test_every_missing_fact_names_a_field_the_context_actually_has() -> None:
    """The drift guard on the moved vocabulary.

    `missing_fact` is validated against `getattr(employee_context, ...)` in the
    scenario schema, so a value here with no matching field would raise at load
    time on some scenarios and pass on others.
    """
    assert set(missing_facts()) == set(EmployeeContext.model_fields)


def test_the_eval_schema_and_the_agent_share_one_vocabulary() -> None:
    """Not two definitions that agree today."""
    import domain
    from eval.scenarios import schema

    for name in ("Route", "Authority", "Jurisdiction", "MissingFact", "EmployeeContext"):
        assert getattr(schema, name) is getattr(domain, name)
