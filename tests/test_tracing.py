"""Tests for the Langfuse export.

No network. `conftest.py` blocks the socket layer, which is also the point of
several of these: the export must be harmless when Langfuse is unreachable.
"""

from __future__ import annotations

from datetime import date

import agent.tracing as tracing
from agent.models import Usage
from agent.state import (
    LayerFinding,
    Resolution,
    TraceEvent,
    VerificationResult,
    initial_state,
)
from agent.tracing import _observation_type, _summarise_input, _summarise_output, run_traced
from domain import EmployeeContext


def sample_state():
    s = initial_state(
        "am I eligible?", EmployeeContext(state="CA", tenure_months=14), date(2026, 4, 1)
    )
    s["route"] = "answer"
    s["citations"] = ["Cal. Gov. Code 12945.2"]
    s["answer"] = "You are eligible [Cal. Gov. Code 12945.2]."
    s["resolution"] = Resolution(
        controlling="state",
        rule="policy_below_floor",
        considered=[
            LayerFinding("state", True, "grants", "Cal. Gov. Code 12945.2", "12mo", 1)
        ],
    )
    s["verification"] = VerificationResult(passed=True)
    s["trace"] = [
        TraceEvent("triage", "answerable", {"route": "answer", "model": "haiku"}),
        TraceEvent("retrieve", "10 passages", {"jurisdiction_filter": "CA"}),
        TraceEvent("resolve", "state controls", {"rule": "policy_below_floor"}),
    ]
    return s


# --- node types map to what Langfuse renders usefully ------------------------


def test_model_calling_nodes_are_generations() -> None:
    """Langfuse surfaces token cost on generations, not on plain spans."""
    for node in ("triage", "resolve", "compose", "verify"):
        assert _observation_type(node) == "generation"


def test_retrieve_is_a_retriever() -> None:
    assert _observation_type("retrieve") == "retriever"


def test_deterministic_nodes_are_plain_spans() -> None:
    """clarify, refuse and escalate call no model, so calling them generations
    would put a zero-token cost row in the UI for work that never had one."""
    for node in ("clarify", "refuse", "escalate"):
        assert _observation_type(node) == "span"


# --- what gets sent ----------------------------------------------------------


def test_the_input_summary_omits_facts_that_were_not_supplied() -> None:
    """A context full of nulls reads as though the asker withheld everything."""
    supplied = _summarise_input(sample_state())["employee_context"]
    assert supplied == {"state": "CA", "tenure_months": 14}


def test_the_output_summary_carries_the_precedence_decision() -> None:
    """Which rule fired is the single most useful field in the whole trace: it
    is the one that explains the product to a non-technical reader."""
    out = _summarise_output(sample_state())
    assert out["controlling_authority"] == "state"
    assert out["precedence_rule"] == "policy_below_floor"
    assert out["verified"] is True


def test_a_run_with_no_resolution_summarises_without_raising() -> None:
    """Refusals and clarifications never resolve an authority."""
    s = initial_state("how do I change my 401k?", as_of=date(2026, 4, 1))
    s["route"] = "refuse"
    out = _summarise_output(s)
    assert out["controlling_authority"] is None
    assert out["verified"] is False


# --- failure is never fatal --------------------------------------------------


def test_export_returns_none_when_langfuse_is_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "configured", lambda: False)
    assert tracing.export(sample_state()) is None


def test_export_swallows_a_broken_client(monkeypatch) -> None:
    """Observability that takes the system down with it is worse than none."""

    class Exploding:
        def start_as_current_observation(self, **kw):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_client", lambda: Exploding())
    assert tracing.export(sample_state()) is None


def test_a_client_that_cannot_be_built_is_not_an_error(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "configured", lambda: True)
    monkeypatch.setattr(
        tracing, "_client", lambda: (_ for _ in ()).throw(RuntimeError("no creds"))
    )
    try:
        tracing.export(sample_state())
    except RuntimeError as exc:
        raise AssertionError(
            "export must not propagate a client construction failure"
        ) from exc


# --- the request boundary ----------------------------------------------------


def test_run_traced_returns_the_state_even_when_export_fails(monkeypatch) -> None:
    """The answer the user asked for must survive a tracing outage."""
    monkeypatch.setattr(tracing, "_client", lambda: None)

    class Graph:
        def invoke(self, state):
            return {**state, "route": "answer"}

    final, url = run_traced(Graph(), sample_state())
    assert final["route"] == "answer"
    assert url is None


def test_run_traced_times_the_invocation(monkeypatch) -> None:
    """Timing is measured at the boundary because the state trace records what
    happened, not when. Splitting a total across nodes would be fabrication."""
    seen = {}

    def fake_export(state, usage=None, elapsed_ms=None, session_id=None):
        seen["elapsed_ms"] = elapsed_ms
        return "url"

    monkeypatch.setattr(tracing, "export", fake_export)

    class Graph:
        def invoke(self, state):
            return state

    _, url = run_traced(Graph(), sample_state())
    assert url == "url"
    assert seen["elapsed_ms"] is not None and seen["elapsed_ms"] >= 0


def test_usage_is_passed_through_for_cost_reporting(monkeypatch) -> None:
    seen = {}

    def fake_export(state, usage=None, elapsed_ms=None, session_id=None):
        seen["usage"] = usage
        seen["session_id"] = session_id
        return None

    monkeypatch.setattr(tracing, "export", fake_export)

    class Graph:
        def invoke(self, state):
            return state

    usage = Usage()
    usage.add("claude-haiku-4-5-20251001", 5400, 260)
    run_traced(Graph(), sample_state(), usage=usage, session_id="s-1")
    assert seen["usage"].input_tokens == 5400
    assert seen["session_id"] == "s-1"
