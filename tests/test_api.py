"""Tests for the API surface.

The agent graph is stubbed. These test the contract the API makes with a caller,
not the agent's answers, which have their own evals. Building the real graph here
would need Qdrant and a funded key, and would test the wrong thing.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from agent.models import Usage
from agent.state import LayerFinding, Resolution, TraceEvent, VerificationResult
from api import precomputed
from api.app import client_ip, create_app, serialise
from api.limits import Protection


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class StubGraph:
    """Answers instantly and records what it was asked."""

    def __init__(self, route: str = "answer"):
        self.route = route
        self.calls: list[dict] = []

    def invoke(self, state):
        self.calls.append(state)
        return {
            **state,
            "route": self.route,
            "answer": "You are eligible [Cal. Gov. Code 12945.2].",
            "citations": ["Cal. Gov. Code 12945.2"],
            "resolution": Resolution(
                controlling="state",
                rule="policy_below_floor",
                considered=[
                    LayerFinding("state", True, "grants", "Cal. Gov. Code 12945.2", "", 1)
                ],
            ),
            "verification": VerificationResult(passed=True),
            "trace": [TraceEvent("triage", "answerable", {"route": "answer"})],
        }


@pytest.fixture
def client(monkeypatch):
    clock = Clock()
    protection = Protection(
        max_input_chars=100,
        per_ip_hourly=3,
        per_session_daily=5,
        global_daily=8,
        now=clock,
    )
    app = create_app(protection)
    agent, baseline = StubGraph(), StubGraph()

    # Never build the real graph here: it needs Qdrant and a funded key, and
    # would test the agent rather than the API contract.
    # Pre-populating `agents` makes lifespan skip the real build, which needs
    # Qdrant and a funded key and would test the agent rather than the contract.
    app.state.agents = {"agent": agent, "baseline": baseline}
    app.state.caller = type("StubCaller", (), {"usage": Usage()})()

    with TestClient(app) as c:
        c.clock = clock
        c.protection = protection
        c.agent = agent
        c.baseline = baseline
        yield c


# --- the free path ----------------------------------------------------------


def test_the_curated_scenarios_are_listed() -> None:
    """These are the six the spec names, and each maps to a scenario in the
    golden set so the demo cannot drift from what is measured."""
    assert set(precomputed.available()) == set(precomputed.CURATED)


def test_a_precomputed_scenario_costs_nothing_and_is_marked_as_such(client) -> None:
    r = client.get("/api/scenario/conflict")
    assert r.status_code == 200
    body = r.json()
    assert body["precomputed"] is True
    assert body["cost_usd"] == 0.0


def test_precomputed_scenarios_bypass_every_limit(client) -> None:
    """The path most reviewers take must work when the budget is gone. Rate
    limiting free work protects a budget it never touches."""
    for _ in range(50):
        client.protection.record("1.1.1.1", "s")
    assert client.protection.remaining_global() == 0
    assert client.get("/api/scenario/conflict").status_code == 200


def test_a_precomputed_scenario_carries_its_ground_truth(client) -> None:
    """A demo that shows only successes is a brochure. The scored expectation
    sits beside the actual result so a reviewer can see which is which."""
    body = client.get("/api/scenario/conflict").json()
    assert body["expected"]["route"] == "answer"
    assert "matched_expectation" in body


def test_an_unknown_scenario_key_is_a_404_not_a_crash(client) -> None:
    assert client.get("/api/scenario/nope").status_code == 404


def test_health_reports_the_limits_and_staleness(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["limits"]["global_daily_limit"] == 8
    assert body["precomputed_stale"] == []


# --- the metered path -------------------------------------------------------


def test_a_question_reaches_the_agent_and_comes_back_serialised(client) -> None:
    r = client.post("/api/ask", json={"question": "am I eligible?", "session_id": "s"})
    assert r.status_code == 200
    body = r.json()
    assert body["controlling_authority"] == "state"
    assert body["precedence_rule"] == "policy_below_floor"
    assert body["precomputed"] is False
    assert client.agent.calls, "the agent was not invoked"


def test_the_baseline_flag_routes_to_the_other_graph(client) -> None:
    """Phase 9's argument in one flag: same graph, precedence swapped out."""
    client.post(
        "/api/ask",
        json={"question": "am I eligible?", "session_id": "s", "baseline": True},
    )
    assert client.baseline.calls and not client.agent.calls


def test_an_empty_question_is_rejected_without_charging_budget(client) -> None:
    before = client.protection.remaining_global()
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400
    assert client.protection.remaining_global() == before


def test_an_overlong_question_is_413_and_charges_nothing(client) -> None:
    before = client.protection.remaining_global()
    r = client.post("/api/ask", json={"question": "x" * 200, "session_id": "s"})
    assert r.status_code == 413
    assert r.json()["limit"] == "input_length"
    assert client.protection.remaining_global() == before


def test_budget_is_charged_only_after_the_work_runs(client) -> None:
    before = client.protection.remaining_global()
    client.post("/api/ask", json={"question": "am I eligible?", "session_id": "s"})
    assert client.protection.remaining_global() == before - 1


def test_the_per_ip_limit_returns_429_with_retry_after(client) -> None:
    for _ in range(3):
        client.post("/api/ask", json={"question": "hello", "session_id": "s"})
    r = client.post("/api/ask", json={"question": "hello", "session_id": "s"})
    assert r.status_code == 429
    assert r.json()["limit"] == "per_ip_hourly"
    assert int(r.headers["Retry-After"]) > 0


def test_a_refusal_tells_the_caller_the_free_path_still_works(client) -> None:
    for _ in range(3):
        client.post("/api/ask", json={"question": "hello", "session_id": "s"})
    body = client.post("/api/ask", json={"question": "hello", "session_id": "s"}).json()
    assert body["precomputed_scenarios_are_always_available"] is True


def test_a_rate_limited_request_never_reaches_the_agent(client) -> None:
    """The whole point: nothing that costs money runs before the gate says so."""
    for _ in range(3):
        client.post("/api/ask", json={"question": "hello", "session_id": "s"})
    calls = len(client.agent.calls)
    client.post("/api/ask", json={"question": "hello", "session_id": "s"})
    assert len(client.agent.calls) == calls


def test_the_limit_clears_when_the_window_slides(client) -> None:
    for _ in range(3):
        client.post("/api/ask", json={"question": "hello", "session_id": "s"})
    assert client.post("/api/ask", json={"question": "hi", "session_id": "s"}).status_code == 429
    client.clock.advance(3601)
    assert client.post("/api/ask", json={"question": "hi", "session_id": "s"}).status_code == 200


# --- client identity --------------------------------------------------------


def test_the_forwarded_header_is_preferred_behind_a_proxy() -> None:
    """Behind Fly.io the socket peer is the proxy, so without this every caller
    shares one bucket and the per-IP limit is a global one under another name."""

    class Req:
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_ip(Req(), "203.0.113.5, 10.0.0.1") == "203.0.113.5"


def test_only_the_first_forwarded_entry_is_trusted() -> None:
    """Later entries are attacker controlled: appending a fake hop is how a
    client would hand itself a fresh bucket."""

    class Req:
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_ip(Req(), "203.0.113.5, 1.2.3.4, 5.6.7.8") == "203.0.113.5"


def test_a_missing_header_falls_back_to_the_socket_peer() -> None:
    class Req:
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_ip(Req(), None) == "10.0.0.1"


def test_an_empty_forwarded_header_does_not_produce_an_empty_key() -> None:
    """An empty bucket key would put every such caller in one shared bucket."""

    class Req:
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_ip(Req(), "  ") == "10.0.0.1"


# --- serialisation ----------------------------------------------------------


def test_serialise_reports_zero_cost_for_a_run_that_called_nothing() -> None:
    state = {"trace": [], "citations": []}
    body = serialise(state, Usage(), None)
    assert body["cost_usd"] == 0.0
    assert body["tokens"] == {"input": 0, "output": 0}


def test_serialise_carries_the_trace_for_the_demo_panel() -> None:
    state = {
        "trace": [TraceEvent("resolve", "state law controls", {"rule": "x"})],
        "citations": [],
    }
    body = serialise(state, Usage(), None)
    assert body["trace"][0]["summary"] == "state law controls"


def test_serialise_survives_a_run_with_no_resolution() -> None:
    body = serialise({"trace": [], "citations": [], "route": "refuse"}, Usage(), None)
    assert body["controlling_authority"] is None
    assert body["verification"]["passed"] is False


def test_precomputed_records_declare_their_provenance() -> None:
    """A demo showing last week's reasoning as current is worse than one that
    admits it needs regenerating."""
    record = precomputed.load("conflict")
    assert record and record.provenance["triage_prompt"]
    assert precomputed.stale(precomputed.current_provenance(date(2026, 8, 1))) == []


def test_a_provenance_mismatch_is_reported_as_stale() -> None:
    drifted = precomputed.stale({"triage_prompt": "triage-v99"})
    assert set(drifted) == set(precomputed.available())
