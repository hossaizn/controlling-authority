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
from agent.tracing import (
    _observation_type,
    _summarise_input,
    _summarise_output,
    run_traced,
    verification_status,
)
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
    assert out["verification"] == "passed"


def test_a_run_with_no_resolution_summarises_without_raising() -> None:
    """Refusals and clarifications never resolve an authority."""
    s = initial_state("how do I change my 401k?", as_of=date(2026, 4, 1))
    s["route"] = "refuse"
    out = _summarise_output(s)
    assert out["controlling_authority"] is None
    assert out["verification"] == "not_applicable"


# --- verification has three states, not two ---------------------------------


def test_a_refusal_reads_as_not_applicable_rather_than_failed() -> None:
    """The DL-26 pattern, reintroduced here and caught by review: a boolean
    reports the same False for "verify ran and failed" and "verify never ran",
    so every correct refusal read as an ungrounded answer."""
    s = initial_state("how do I change my 401k?", as_of=date(2026, 4, 1))
    s["route"] = "refuse"
    assert verification_status(s) == "not_applicable"


def test_an_answering_path_that_skipped_verify_reads_as_did_not_run() -> None:
    """A defect worth seeing in the trace, not laundered into a failure."""
    s = initial_state("how much leave?", as_of=date(2026, 4, 1))
    s["route"] = "answer"
    assert verification_status(s) == "did_not_run"


def test_a_failed_verification_is_distinguishable_from_both() -> None:
    s = sample_state()
    s["verification"] = VerificationResult(passed=False, failures=["ungrounded"])
    assert verification_status(s) == "failed"


def test_an_indeterminate_resolution_does_not_read_as_no_authority() -> None:
    """`controlling` is None on a legitimate concurrence tie. Without the
    defensible set, that renders identically to "nothing controls"."""
    s = sample_state()
    s["resolution"] = Resolution(
        controlling=None, rule="indeterminate", acceptable=["federal", "state"]
    )
    out = _summarise_output(s)
    assert out["controlling_authority"] is None
    assert out["defensible_authorities"] == ["federal", "state"]


def test_the_input_summary_records_what_actually_reached_the_index() -> None:
    """The module claims to mirror the run. Dropping the rewritten query, the
    jurisdiction filter and the hit list made that claim false."""
    s = sample_state()
    s["rewritten_query"] = "parental leave eligibility"
    s["jurisdiction"] = "CA"
    got = _summarise_input(s)
    assert got["query_sent_to_index"] == "parental leave eligibility"
    assert got["jurisdiction_filter"] == "CA"
    assert got["passages_retrieved"] == []


# --- failure is never fatal --------------------------------------------------


def test_export_returns_none_when_langfuse_is_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_DEBUG", raising=False)
    monkeypatch.setattr(tracing, "configured", lambda: False)
    assert tracing.export(sample_state()) is None


def test_export_swallows_a_broken_client(monkeypatch) -> None:
    """Observability that takes the system down with it is worse than none."""
    monkeypatch.delenv("LANGFUSE_DEBUG", raising=False)

    class Exploding:
        def start_as_current_observation(self, **kw):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_client", lambda: Exploding())
    assert tracing.export(sample_state()) is None


def test_a_client_that_cannot_be_built_is_not_an_error(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_DEBUG", raising=False)
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


def test_timed_measures_real_elapsed_time() -> None:
    """`>= 0` is a relationship on a derived value and a hardcoded 0.0 satisfies
    it. DL-10's finding, in this repo's own non-negotiables: for derived values,
    assert the value. A 25ms sleep must show up as at least 20ms."""
    import time as _time

    from agent.tracing import timed

    with timed() as clock:
        _time.sleep(0.025)
    assert clock["elapsed_ms"] >= 20.0


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


# --- export's body, which had no test at all ---------------------------------
#
# A review mutated it 25 ways and 19 survived: dropping the entire child-event
# loop, swapping child input/output, never sending usage, never flushing, never
# attaching session_id, exporting the pre-invoke state. The failure paths were
# tested and the thing the module exists to do was not. Same finding as the
# Phase 6 review, one module later.


class RecordingSpan:
    def __init__(self, name, sink, /, **kw):
        self.name = name
        self.sink = sink
        self.kw = kw
        self.updates: list[dict] = []
        self.ended = False

    def start_observation(self, **kw):
        child = RecordingSpan(kw.get("name"), self.sink, **kw)
        self.sink["children"].append(child)
        return child


    def update(self, **kw):
        self.updates.append(kw)

    def end(self):
        self.ended = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingClient:
    def __init__(self):
        self.sink = {"children": [], "root": None, "flushed": False}

    def start_as_current_observation(self, **kw):
        root = RecordingSpan(kw.get("name"), self.sink, **kw)
        self.sink["root"] = root
        return root

    def get_trace_url(self):
        return "https://langfuse.example/trace/abc"

    def flush(self):
        self.sink["flushed"] = True


def record(state, monkeypatch, **kw):
    client = RecordingClient()
    monkeypatch.setattr(tracing, "_client", lambda: client)
    url = tracing.export(state, **kw)
    return client.sink, url


def test_every_trace_event_becomes_one_observation(monkeypatch) -> None:
    """Dropping the child loop entirely left the suite green."""
    sink, _ = record(sample_state(), monkeypatch)
    assert [c.name for c in sink["children"]] == ["triage", "retrieve", "resolve"]


def test_each_observation_carries_the_summary_as_input_and_detail_as_output(
    monkeypatch,
) -> None:
    """Swapping these two survived mutation. The summary is the human-readable
    line; the detail is the structured record. Reversed, the UI is unreadable."""
    sink, _ = record(sample_state(), monkeypatch)
    triage = sink["children"][0]
    assert triage.kw["input"] == {"summary": "answerable"}
    assert triage.kw["output"]["route"] == "answer"


def test_every_observation_is_ended(monkeypatch) -> None:
    sink, _ = record(sample_state(), monkeypatch)
    assert all(c.ended for c in sink["children"])


def test_model_calling_nodes_are_typed_as_generations_in_the_export(
    monkeypatch,
) -> None:
    sink, _ = record(sample_state(), monkeypatch)
    by_name = {c.name: c.kw["as_type"] for c in sink["children"]}
    assert by_name["triage"] == "generation"
    assert by_name["retrieve"] == "retriever"


def test_a_node_with_no_model_gets_no_model_metadata(monkeypatch) -> None:
    """`metadata={"model": None}` on every deterministic node invents a field
    the run does not have."""
    sink, _ = record(sample_state(), monkeypatch)
    retrieve = next(c for c in sink["children"] if c.name == "retrieve")
    assert retrieve.kw["metadata"] is None


def test_the_root_receives_the_output_summary(monkeypatch) -> None:
    """Deleting this update survived. Without it the trace has no verdict."""
    sink, _ = record(sample_state(), monkeypatch)
    outputs = [u["output"] for u in sink["root"].updates if "output" in u]
    assert outputs and outputs[0]["precedence_rule"] == "policy_below_floor"


def test_usage_and_cost_reach_the_root(monkeypatch) -> None:
    usage = Usage()
    usage.add("claude-haiku-4-5-20251001", 5400, 260)
    sink, _ = record(sample_state(), monkeypatch, usage=usage)
    details = [u for u in sink["root"].updates if "usage_details" in u]
    assert details and details[0]["usage_details"] == {"input": 5400, "output": 260}


def test_usage_with_no_calls_is_not_reported(monkeypatch) -> None:
    """An all-cache run spent nothing. Reporting zero tokens as a measurement
    would put a real-looking $0.00 next to work that never called a model."""
    sink, _ = record(sample_state(), monkeypatch, usage=Usage())
    assert not [u for u in sink["root"].updates if "usage_details" in u]


def test_elapsed_time_reaches_the_root(monkeypatch) -> None:
    sink, _ = record(sample_state(), monkeypatch, elapsed_ms=13198.04)
    meta = [u["metadata"] for u in sink["root"].updates if "metadata" in u]
    assert any(m.get("elapsed_ms") == 13198.0 for m in meta)


def test_the_session_id_is_attached(monkeypatch) -> None:
    sink, _ = record(sample_state(), monkeypatch, session_id="s-42")
    assert sink["root"].kw["metadata"] == {"session_id": "s-42"}


def test_the_client_is_flushed(monkeypatch) -> None:
    """Langfuse batches. Without a flush a short-lived process exports nothing,
    and the failure is invisible because export still returns a URL."""
    sink, _ = record(sample_state(), monkeypatch)
    assert sink["flushed"] is True


def test_the_trace_url_is_returned(monkeypatch) -> None:
    _, url = record(sample_state(), monkeypatch)
    assert url == "https://langfuse.example/trace/abc"


def test_run_traced_exports_the_state_after_invocation(monkeypatch) -> None:
    """Exporting the pre-invoke state survived mutation: it produces a trace
    with no route, no answer and no events, which looks like a quiet run."""
    client = RecordingClient()
    monkeypatch.setattr(tracing, "_client", lambda: client)

    class Graph:
        def invoke(self, state):
            return {
                **state,
                "route": "answer",
                "trace": [*state["trace"], TraceEvent("verify", "4/4 passed", {})],
            }

    final, _ = run_traced(Graph(), sample_state())
    assert final["route"] == "answer"
    assert [c.name for c in client.sink["children"]][-1] == "verify"


# --- privacy: TRACE_REDACT ---------------------------------------------------
#
# A review found this undocumented, unmentioned and undisableable: the question,
# the employee's tenure, hours, employer size and state, and the full answer all
# went to a third-party SaaS with no way to stop it short of unsetting the keys.


def test_redaction_removes_the_question_and_the_answer(monkeypatch) -> None:
    monkeypatch.setenv("TRACE_REDACT", "1")
    assert _summarise_input(sample_state())["question"] == tracing.REDACTED
    assert _summarise_output(sample_state())["answer"] == tracing.REDACTED


def test_redaction_removes_personal_values_but_keeps_which_facts_were_supplied(
    monkeypatch,
) -> None:
    """Which facts the asker gave still explains a clarify decision. Their
    values are what identify a person."""
    monkeypatch.setenv("TRACE_REDACT", "1")
    got = _summarise_input(sample_state())
    assert "employee_context" not in got
    assert got["employee_context_supplied"] == ["state", "tenure_months"]


def test_redaction_keeps_everything_needed_to_debug_the_system(monkeypatch) -> None:
    """The point is a trace that is still useful. Routes, precedence rules,
    citations and filters are structural, not personal."""
    monkeypatch.setenv("TRACE_REDACT", "1")
    out = _summarise_output(sample_state())
    assert out["precedence_rule"] == "policy_below_floor"
    assert out["citations"] == ["Cal. Gov. Code 12945.2"]
    assert out["verification"] == "passed"
    assert _summarise_input(sample_state())["jurisdiction_filter"] is None


def test_redaction_scrubs_free_text_inside_node_details(monkeypatch) -> None:
    """`raw_question` in triage's detail and `says` in resolve's carry the
    question and quoted passage text straight through."""
    monkeypatch.setenv("TRACE_REDACT", "1")
    client = RecordingClient()
    monkeypatch.setattr(tracing, "_client", lambda: client)
    s = sample_state()
    s["trace"] = [
        TraceEvent("triage", "answerable", {"raw_question": "am I eligible?", "route": "answer"})
    ]
    tracing.export(s)
    detail = client.sink["children"][0].kw["output"]
    assert detail["raw_question"] == tracing.REDACTED
    assert detail["route"] == "answer"


def test_redaction_is_off_by_default(monkeypatch) -> None:
    """Opt-in on purpose: this is a demo with no real employees, and defaulting
    it on would hide the trace the demo exists to show. A real deployment turns
    it on or self-hosts."""
    monkeypatch.delenv("TRACE_REDACT", raising=False)
    assert tracing.redacting() is False
    assert _summarise_input(sample_state())["question"] == "am I eligible?"
