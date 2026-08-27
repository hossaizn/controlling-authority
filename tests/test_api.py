"""Tests for the API surface.

The agent graph is stubbed. These test the contract the API makes with a caller,
not the agent's answers, which have their own evals. Building the real graph here
would need Qdrant and a funded key, and would test the wrong thing.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

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



def hit_for(citation, layer="federal"):
    from retrieval.store import SearchHit

    return SearchHit(
        chunk_id=f"c-{citation}", citation=citation, authority_layer=layer,
        jurisdiction="US", content_status="substantive", heading="h",
        text="t", score=0.5,
    )


def _baseline_record(state, expected):
    """Drive `save_baseline` and read back what it wrote.

    A review mutated `correct` to True and dropped both expectation fields, and
    every mutation survived: the tests only read committed JSON, never the
    function that produces it.
    """
    import json as _json
    from unittest.mock import patch

    # **Never writes near the real records.** An earlier version wrote over the
    # committed conflict baseline and restored it in a `finally` that was
    # initially a no-op; the damage then became self-perpetuating, because the
    # fixed version captured the polluted file as its "original". The tell was
    # the test's own fake provenance sitting in committed demo data.
    #
    # Redirecting STORE removes the possibility rather than managing it.
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(precomputed, "STORE", Path(tmp)):
            path = precomputed.save_baseline("conflict", state, {"p": "v"}, expected)
            return _json.loads(path.read_text())


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


def test_the_rightmost_forwarded_entry_is_trusted() -> None:
    """**This was backwards and exploitable.** Each proxy APPENDS, so the
    leftmost entry is whatever the client sent and the rightmost was written by
    our own edge. Trusting the left one let a caller rotate the header and mint
    a fresh rate-limit bucket per request."""

    class Req:
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_ip(Req(), "1.2.3.4, 203.0.113.5") == "203.0.113.5"


def test_a_spoofed_leading_entry_cannot_choose_the_bucket() -> None:
    class Req:
        client = type("C", (), {"host": "10.0.0.1"})()

    a = client_ip(Req(), "attacker-picked-1, 203.0.113.5")
    b = client_ip(Req(), "attacker-picked-2, 203.0.113.5")
    assert a == b == "203.0.113.5", "rotating the client-supplied hop must not help"


def test_the_platform_header_wins_over_forwarded_for() -> None:
    """Fly-Client-IP is set by the platform and cannot be forwarded by a client,
    so it is preferred over anything in X-Forwarded-For."""

    class Req:
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_ip(Req(), "1.2.3.4, 5.6.7.8", "203.0.113.9") == "203.0.113.9"


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


# --- the review's confirmed bugs, pinned ------------------------------------


def test_a_failing_agent_still_charges_budget(client) -> None:
    """**The worst bug in Phase 8.** Recording only on success meant a graph
    that raised after calling the model spent real tokens and consumed no
    budget: 30 such requests spent 3,000 tokens with the global counter still
    reading 8 of 8. The breaker was not the last line, it was only reached by
    requests that succeeded."""

    class Exploding:
        def invoke(self, state):
            raise RuntimeError("boom, after the model was called")

    client.app.state.agents["agent"] = Exploding()
    before = client.protection.remaining_global()
    with pytest.raises(RuntimeError):
        client.post("/api/ask", json={"question": "hello", "session_id": "s"})
    assert client.protection.remaining_global() == before - 1


def test_repeated_failures_are_still_capped_by_the_breaker(client) -> None:
    class Exploding:
        def invoke(self, state):
            raise RuntimeError("boom")

    client.app.state.agents["agent"] = Exploding()
    for _ in range(3):
        with pytest.raises(RuntimeError):
            client.post("/api/ask", json={"question": "hi", "session_id": "s"})
    r = client.post("/api/ask", json={"question": "hi", "session_id": "s"})
    assert r.status_code == 429


def test_slice_performance_is_not_empty(client) -> None:
    """It read a gitignored directory, so it was `{}` in every deployment and
    DL-29's honesty argument held only on the machine that generated it. It
    failed silently: a missing file returned `{}` and all tests passed."""
    body = client.get("/api/scenario/conflict").json()
    assert body["slice_performance"], "slice scores must ship with the records"
    assert body["slice_performance"]["n"] == 18
    assert 0.0 < body["slice_performance"]["fully_correct"] < 1.0


def test_every_payload_carries_its_slice_scores(client) -> None:
    for key in precomputed.available():
        body = client.get(f"/api/scenario/{key}").json()
        assert body["slice_performance"], f"{key} has no slice scores"


def test_the_slice_snapshot_is_committed_not_gitignored() -> None:
    """The bug was that the source was ignored by git. Pin the file's presence
    so it cannot quietly stop shipping again."""
    assert precomputed.SCORES.exists()


def test_a_scenario_response_declares_whether_it_is_stale(client, monkeypatch) -> None:
    """`precomputed.py` says a stale record must not be served as though it were
    current, and only /api/health honoured that.

    Asserting only `is False` let a mutation that hardcodes False survive, so
    both directions are checked.
    """
    assert client.get("/api/scenario/conflict").json()["stale"] is False
    monkeypatch.setattr(precomputed, "stale", lambda _: ["conflict"])
    assert client.get("/api/scenario/conflict").json()["stale"] is True


def test_a_partial_provenance_drift_is_stale() -> None:
    """`any`, not `all`: one changed prompt makes a record stale even if the
    corpus snapshot still matches. Requiring every field to differ would mean a
    record is only stale when nothing about it is current."""
    current = precomputed.current_provenance(date(2026, 8, 1))
    drifted = {**current, "resolve_prompt": "resolve-v99"}
    assert set(precomputed.stale(drifted)) == set(precomputed.available())


def test_matched_expectation_is_false_when_the_run_missed() -> None:
    """Asserting only that the key exists let a mutation hardcoding True
    survive, which would make every scenario look correct."""
    record = precomputed.load("conflict")
    assert record.matched_expectation is True
    missed = replace(record, route="refuse")
    assert missed.matched_expectation is False


def test_the_route_shown_is_the_one_the_run_produced() -> None:
    """Serving `expected["route"]` instead would make the demo display ground
    truth as though it were the agent's output: every scenario correct forever."""
    record = precomputed.load("conflict")
    faked = replace(record, route="refuse", expected={**record.expected, "route": "answer"})
    assert faked.payload["route"] == "refuse"
    assert faked.payload["matched_expectation"] is False


def test_an_unknown_key_cannot_escape_the_store() -> None:
    """`CURATED` is the allowlist and the only traversal guard: without it, a
    key is interpolated straight into a path."""
    for key in ("../../etc/passwd", "../../../CLAUDE", "..%2f..%2fsecrets"):
        assert precomputed.load(key) is None


def test_a_key_outside_the_allowlist_is_refused_even_if_a_file_exists(tmp_path) -> None:
    import json as _json

    from api import precomputed as pc

    rogue = pc.STORE / "rogue.json"
    rogue.write_text(_json.dumps({"scenario_id": "x", "question": "q", "as_of": "d",
                                  "answer": "a", "route": "answer"}))
    try:
        assert pc.load("rogue") is None, "only curated keys may be served"
    finally:
        rogue.unlink()


def test_expired_keys_are_evicted_rather_than_accumulating() -> None:
    """`<= 500` was the original assertion and a budget that never evicted also
    satisfies it: it asserted a relationship, not a value (DL-10).

    Session id and forwarded IP are client-supplied and unauthenticated, so a
    caller can mint unlimited keys; 500 refused requests allocated 1,000
    permanent deques before eviction existed.
    """
    from api.limits import SlidingWindow

    clock = Clock()
    window = SlidingWindow(60.0, clock)
    window.MAX_KEYS = 5
    for i in range(50):
        window.record(f"k{i}")
    clock.advance(61)  # every key is now expired
    window.record("trigger")
    assert len(window._events) == 1, "expired keys must be dropped, not kept"


def test_eviction_never_forgives_a_caller_inside_their_window() -> None:
    """Growing memory is preferable to handing out free budget."""
    from api.limits import SlidingWindow

    clock = Clock()
    window = SlidingWindow(60.0, clock)
    window.MAX_KEYS = 2
    for i in range(10):
        window.record(f"k{i}")
    assert window.count("k0") == 1, "a live key was evicted"


def test_the_global_key_survives_eviction_even_when_expired() -> None:
    """Eviction bounds memory; it must never reach the budget.

    The first version never advanced the clock, so "all" was live and no
    eviction could have touched it either way: the test passed with the
    protection removed.
    """
    from api.limits import SlidingWindow

    clock = Clock()
    window = SlidingWindow(60.0, clock)
    window.MAX_KEYS = 2
    window.record("all")
    clock.advance(61)
    for i in range(10):
        window.record(f"noise{i}")
    assert "all" in window._events, "the budget key must never be evicted"


# --- the demo (Phase 9) ------------------------------------------------------


def test_the_page_is_served_and_self_contained(client) -> None:
    """No CDN, no build step. Every dependency is another thing that can be
    unavailable when someone opens it."""
    import re

    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "<title>Controlling Authority</title>" in html

    # Anything that would load from another origin at render time. Matching on
    # the word "cdn" tripped on this file's own comment saying it uses none,
    # which is the shape of test that checks the prose rather than the code.
    external = re.findall(
        r'(?:src|href)\s*=\s*["\']((?:https?:)?//[^"\']+)', html, re.I
    )
    assert external == [], f"page loads external resources: {external}"
    assert not re.search(r"url\(\s*[\"\']?(?:https?:)?//", html, re.I)
    assert "import(" not in html


def test_the_page_never_builds_html_from_data() -> None:
    """An escape helper covering &<> is one refactor away from being wrong, and
    this endpoint is public. Values reach the DOM via textContent or a created
    node."""
    html = (Path(__file__).resolve().parent.parent / "api/static/index.html").read_text()
    assert ".innerHTML" not in html


def test_the_baseline_arm_is_served_for_every_scenario(client) -> None:
    """The toggle is the demo's central comparison; it cannot be the one screen
    that needs a funded key."""
    for key in precomputed.available():
        r = client.get(f"/api/scenario/{key}/baseline")
        assert r.status_code == 200, key


def test_the_baseline_arm_records_no_answer_only_a_resolution(client) -> None:
    """Composing a baseline answer would add two model calls per scenario to
    restate a conclusion the resolution already shows."""
    b = client.get("/api/scenario/conflict/baseline").json()
    assert "answer" not in b
    assert b["controlling_authority"]
    assert "top-ranked passage" in b["precedence_rule"]


def test_the_conflict_scenario_shows_a_real_delta(client) -> None:
    """If the agent and the naive baseline agreed here, the demo's central
    screen would be blank. conflict-002 is chosen precisely because they do
    not: the handbook exceeds the statutory floor, so policy controls, and a
    system that trusts the closest passage picks the statute."""
    a = client.get("/api/scenario/conflict").json()
    b = client.get("/api/scenario/conflict/baseline").json()
    assert a["controlling_authority"] == "company"
    assert b["controlling_authority"] == "state"
    assert b["correct"] is False
    assert a["matched_expectation"] is True


def test_the_agent_record_carries_its_resolution(client) -> None:
    """Without it the baseline comparison has nothing to compare."""
    a = client.get("/api/scenario/conflict").json()
    assert a["precedence_rule"] == "policy_may_exceed"
    assert "defensible_authorities" in a


def test_the_supersession_pair_is_the_same_question_at_two_dates() -> None:
    """The clearest demonstration of point-in-time answering, and the same pair
    that proved in DL-21 that the date cannot be recovered from the question."""
    before, after = (precomputed.load(k) for k in precomputed.SUPERSESSION_PAIR)
    assert before.question == after.question
    assert before.employee_context == after.employee_context
    assert before.as_of != after.as_of
    assert before.answer != after.answer


def test_health_exposes_the_measured_scores_for_the_footer(client) -> None:
    """The page quotes them, so they come from the committed snapshot rather
    than being restated in the markup where they could drift."""
    scores = client.get("/api/health").json()["overall_scores"]
    assert scores["n"] == 92
    assert 0.0 < scores["fully_correct"] < 1.0


def test_an_unknown_baseline_key_is_a_404(client) -> None:
    assert client.get("/api/scenario/nope/baseline").status_code == 404


# --- the baseline arm must describe the graph it claims to represent --------


def test_the_baseline_only_resolves_on_the_answering_path() -> None:
    """**The worst honesty failure in Phase 9.**

    `build_baseline` shares the agent's triage and the graph gates retrieval
    behind the route, so on clarify, refuse and escalate the baseline declines
    identically. The first generator ran retrieve and naive_resolve
    unconditionally and recorded an authority for scenarios the baseline never
    resolves, so the demo asserted a naive system "would have answered" three
    questions it actually refuses, contradicting `/api/ask {baseline: true}`.
    """
    for key in precomputed.available():
        agent = precomputed.load(key)
        base = precomputed.load_baseline(key)
        if agent.route != "answer":
            assert base["resolved"] is False, f"{key}: baseline cannot resolve here"
            assert base["controlling_authority"] is None
            assert base["no_delta_reason"]
        else:
            assert base["resolved"] is True, key


def test_only_one_scenario_currently_shows_a_baseline_delta() -> None:
    """Pinned as a value. DL-31 originally claimed four, which was wrong because
    three of them were describing a code path that never runs. If a future
    change makes more of them differ, this fails and the number gets updated
    deliberately rather than drifting."""
    wrong = [
        k for k in precomputed.available()
        if precomputed.load_baseline(k)["resolved"] and not precomputed.load_baseline(k)["correct"]
    ]
    assert wrong == ["conflict"]


def test_a_declining_baseline_is_not_labelled_wrong(client) -> None:
    b = client.get("/api/scenario/refuse/baseline").json()
    assert b["resolved"] is False
    assert b["correct"] is True, "declining correctly is not a failure"


# --- save_baseline had no coverage at all -----------------------------------


def test_save_baseline_marks_a_matching_authority_correct(tmp_path) -> None:
    state = {
        "route": "answer",
        "resolution": Resolution(controlling="company", rule="not_reached"),
        "retrieved": [hit_for("LEAVE-003", "company")],
    }
    rec = _baseline_record(state, {"authority": "company", "route": "answer"})
    assert rec["correct"] is True


def test_save_baseline_marks_a_mismatched_authority_wrong() -> None:
    state = {
        "route": "answer",
        "resolution": Resolution(controlling="state", rule="not_reached"),
        "retrieved": [hit_for("Cal. Gov. Code 12945.2", "state")],
    }
    rec = _baseline_record(state, {"authority": "company", "route": "answer"})
    assert rec["correct"] is False


def test_save_baseline_accepts_any_member_of_an_indeterminate_set() -> None:
    """Mirrors `eval/run_precedence.acceptable_set`: where the answer is
    determinate and the controlling layer is not, either is correct."""
    state = {
        "route": "answer",
        "resolution": Resolution(controlling="federal", rule="not_reached"),
        "retrieved": [hit_for("29 CFR 825.110")],
    }
    rec = _baseline_record(
        state, {"acceptable_authorities": ["federal", "state"], "route": "answer"}
    )
    assert rec["correct"] is True


def test_save_baseline_with_no_expectation_is_not_correct_by_default() -> None:
    state = {
        "route": "answer",
        "resolution": Resolution(controlling="company", rule="not_reached"),
        "retrieved": [],
    }
    assert _baseline_record(state, {})["correct"] is False


# --- matched_expectation checks more than the route -------------------------


def test_matched_expectation_requires_the_right_authority() -> None:
    """The page paints this green as "matches ground truth". The right
    conclusion from the wrong authority is what the spec calls luck."""
    r = precomputed.load("conflict")
    wrong_authority = replace(r, controlling_authority="state", defensible_authorities=["state"])
    assert wrong_authority.matched_expectation is False


def test_matched_expectation_requires_the_required_citations() -> None:
    r = precomputed.load("conflict")
    assert r.expected["required_citations"]
    assert replace(r, answer="You get some leave.").matched_expectation is False


# --- the page's honesty elements ---------------------------------------------
#
# A review mutated the rendering six ways and every one survived: the baseline
# tag forced to "right, by luck", the agent tag forced true, the footer stats
# blanked, the slice box permanently hidden. Only the PAYLOAD was tested, never
# what the page does with it.
#
# These are structural checks on the source, not render tests. A real render
# test needs a JS runtime, which is a dependency this project does not have and
# does not need for one page. Structural checks catch deletion and inversion of
# the honesty elements, which is what the mutations did; they would not catch a
# subtle CSS change that hides one. That limit is stated rather than papered
# over.


def page_source() -> str:
    return (Path(__file__).resolve().parent.parent / "api/static/index.html").read_text()


def test_the_page_renders_the_measured_overall_score() -> None:
    src = page_source()
    assert "fully_correct" in src
    assert "route_accuracy_macro" in src
    assert "including where it fails" in src


def test_the_page_renders_the_slice_score() -> None:
    """One working example is not a working slice."""
    src = page_source()
    assert "renderSlice" in src
    assert "One working example is not a working slice" in src


def test_the_page_distinguishes_all_three_baseline_outcomes() -> None:
    """wrong, right-by-luck, and no-difference. Collapsing any two is how the
    demo came to assert a delta that does not exist."""
    src = page_source()
    for label in ('"no difference"', '"right, by luck"', '"wrong"'):
        assert label in src, label
    assert "b.resolved === false" in src


def test_the_page_labels_the_agent_against_ground_truth() -> None:
    src = page_source()
    assert "matched_expectation" in src
    assert "differs from ground truth" in src


def test_the_page_surfaces_staleness() -> None:
    src = page_source()
    assert "staleWarn" in src
    assert "may not reflect current behaviour" in src


def test_the_page_handles_a_missing_scores_snapshot() -> None:
    """Otherwise the footer sits on "Loading measured results…" forever."""
    assert "Measured results are not available" in page_source()


def test_the_page_handles_a_non_json_error_response() -> None:
    """A proxy 502 returns HTML; without this the promise rejects and the user
    watches a spinner that never resolves."""
    assert "unexpected response" in page_source()


# --- advisories reach the surface -------------------------------------------


def test_the_api_reports_blocking_and_grounded_separately(client) -> None:
    """`passed` means nothing blocked the answer; `fully_grounded` also requires
    the judgment call. Collapsing them would hide the relaxation."""
    r = client.post("/api/ask", json={"question": "am I eligible?", "session_id": "s"})
    v = r.json()["verification"]
    assert "passed" in v and "fully_grounded" in v and "advisories" in v


def test_a_precomputed_record_carries_its_advisories(client) -> None:
    body = client.get("/api/scenario/conflict").json()
    assert "advisories" in body


def test_the_page_renders_flagged_claims() -> None:
    """An answer with a flagged claim gives the reader the provision, the
    reasoning, and the sentence to be careful about. A referral gives nothing."""
    src = page_source()
    assert "renderFlags" in src
    assert "could not be tied to its cited passage" in src
    assert "aFlags" in src


def test_the_page_labels_a_flagged_answer_as_such() -> None:
    src = page_source()
    assert '"claims flagged"' in src
    assert "fully_grounded" in src
