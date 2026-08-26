"""Langfuse tracing, exported from the trace the agent already keeps.

**The state's `trace` list is the single source of truth, and this mirrors it.**
Instrumenting the nodes a second time, with decorators or context managers, would
create two descriptions of one run that agree until someone edits one of them.
This project has found that failure enough times to stop inviting it: two copies
of the naive baseline (DL-26), two definitions of one tool contract (DL-24), and
`speaks_to_question` duplicating `outcome` (DL-26).

So nodes stay unaware of Langfuse. They append `TraceEvent`s because the trace is
the demo's product, and this walks the finished list afterwards.

**Two consequences worth stating.** The export is a faithful mirror of what the
user sees, so a reviewer reading the Langfuse UI and a reviewer reading the demo
panel cannot be shown different stories. And **an unconfigured or broken Langfuse
cannot break a run**: `trace()` degrades to a no-op, because observability that
takes the system down with it is worse than none.

Per-node timing is not available from the state trace, which records what
happened rather than when. Timing is captured here by the caller wrapping the
invocation, so the spans carry real wall-clock rather than a fabricated split.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from agent.models import Usage
from agent.state import AgentState
from ingest.settings import optional

# --- Privacy -----------------------------------------------------------------
#
# **This module sends employee data to a third-party SaaS.** The question, the
# supplied context (state, tenure, hours worked, employer size) and the final
# answer all leave the system. For an HCM product that is employee PII, and a
# review found it undocumented, unmentioned and undisableable except by unsetting
# the credentials.
#
# `TRACE_REDACT=1` sends the structure without the content: routes, precedence
# rules, citations, filters and timings, but no question, no personal facts and
# no answer text. That keeps the trace useful for debugging the *system* while
# removing what identifies a person.
#
# It is opt-in rather than default because this is a portfolio demo with no real
# employees, and defaulting it on would hide the feature the demo exists to show.
# **Any real deployment should turn it on, or self-host Langfuse.** Recorded here
# rather than in a README because the decision belongs next to the code that
# makes it.
REDACTED = "[redacted]"


def redacting() -> bool:
    return bool(optional("TRACE_REDACT"))


# Nodes that call a model. Langfuse renders these as generations rather than
# plain spans, which is what surfaces token cost in its UI.
_GENERATION_NODES = {"triage", "resolve", "compose", "verify"}

# Retrieval gets its own observation type so the filters and hit list are
# readable as a retrieval step rather than an opaque span.
_RETRIEVER_NODES = {"retrieve"}


def configured() -> bool:
    return bool(optional("LANGFUSE_PUBLIC_KEY") and optional("LANGFUSE_SECRET_KEY"))


def _client():
    """Built lazily and never cached across failure.

    Returns None rather than raising when unconfigured, so importing this module
    in a test or running the agent without credentials costs nothing.
    """
    if not configured():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=optional("LANGFUSE_PUBLIC_KEY"),
            secret_key=optional("LANGFUSE_SECRET_KEY"),
            host=optional("LANGFUSE_HOST") or None,
        )
    except Exception:
        return None


# Keys whose values are free text derived from the question or the answer.
# Everything else in a node detail is structural: routes, rules, citations,
# filters, layer outcomes.
_SENSITIVE_DETAIL_KEYS = {"raw_question", "query", "query_sent_to_index", "says"}


def _redact_detail(detail: Any) -> Any:
    if not isinstance(detail, dict):
        return detail
    return {
        k: (REDACTED if k in _SENSITIVE_DETAIL_KEYS else v) for k, v in detail.items()
    }


def _observation_type(node: str) -> str:
    if node in _GENERATION_NODES:
        return "generation"
    if node in _RETRIEVER_NODES:
        return "retriever"
    return "span"


def _summarise_input(state: AgentState) -> dict[str, Any]:
    context = state.get("employee_context")
    if redacting():
        return {
            "question": REDACTED,
            "as_of": str(state.get("as_of")),
            # Which facts were supplied still matters for debugging a clarify
            # decision; their values do not.
            "employee_context_supplied": sorted(
                k for k, v in (context.model_dump() if context else {}).items()
                if v is not None
            ),
            "query_sent_to_index": REDACTED,
            "jurisdiction_filter": state.get("jurisdiction"),
            "passages_retrieved": [h.citation for h in state.get("retrieved", [])],
        }
    return {
        "question": state.get("question"),
        "as_of": str(state.get("as_of")),
        "employee_context": (
            {k: v for k, v in context.model_dump().items() if v is not None}
            if context
            else {}
        ),
        # What actually reached the index, and under which filters. The module
        # claims to mirror the run; dropping these made that claim false.
        "query_sent_to_index": state.get("rewritten_query"),
        "jurisdiction_filter": state.get("jurisdiction"),
        "passages_retrieved": [h.citation for h in state.get("retrieved", [])],
    }


def verification_status(state: AgentState) -> str:
    """Three states, not two.

    `bool(verification and verification.passed)` reports the same `False` for
    "verify ran and failed" and "verify never ran", which is **exactly the bug
    DL-25 fixed in the end-to-end scorer and DL-26 recorded as a pattern**:
    fixing an instance of a bug is not fixing the bug. It reappeared here.

    `clarify`, `refuse` and `escalate` assert no entitlement, so there is nothing
    to ground and `not_applicable` is the honest label. An answering path that
    somehow skipped verify is `did_not_run`, which is a defect worth seeing in
    the trace rather than laundering into a failure.
    """
    verification = state.get("verification")
    if verification is not None:
        return "passed" if verification.passed else "failed"
    if state.get("route") in ("clarify", "refuse", "escalate"):
        return "not_applicable"
    return "did_not_run"


def _summarise_output(state: AgentState) -> dict[str, Any]:
    verification = state.get("verification")
    resolution = state.get("resolution")
    return {
        "route": state.get("route"),
        # `controlling` is None on a legitimate indeterminate tie, so the
        # defensible set is reported too: without it, `concurrence_tie_break`
        # renders identically to "no authority found".
        "controlling_authority": resolution.controlling if resolution else None,
        "defensible_authorities": list(resolution.defensible) if resolution else [],
        "precedence_rule": resolution.rule if resolution else None,
        "considered": (
            [
                {"layer": f.layer, "outcome": f.outcome, "citation": f.citation}
                for f in resolution.considered
            ]
            if resolution
            else []
        ),
        "must_address": (
            list(resolution.non_controlling_to_address) if resolution else []
        ),
        "citations": state.get("citations", []),
        "verification": verification_status(state),
        "verification_checks": dict(verification.checks) if verification else {},
        "verification_failures": list(verification.failures) if verification else [],
        "answer": REDACTED if redacting() else state.get("answer"),
    }


def export(
    state: AgentState,
    usage: Usage | None = None,
    elapsed_ms: float | None = None,
    session_id: str | None = None,
) -> str | None:
    """Mirror a finished run's trace into Langfuse. Returns the trace URL.

    Never raises. A failed export is reported as `None` and the answer the user
    asked for is unaffected.
    """
    # Client construction is inside the try, not before it. `_client` guards
    # itself, but then "export never raises" holds only as long as two separate
    # places agree about that, and a test written against the contract caught the
    # gap immediately. The guarantee belongs where the contract is stated.
    try:
        client = _client()
        if client is None:
            return None

        with client.start_as_current_observation(
            name="controlling-authority",
            as_type="agent",
            input=_summarise_input(state),
            metadata={"session_id": session_id} if session_id else None,
        ) as root:
            for event in state.get("trace", []):
                child = root.start_observation(
                    name=event.node,
                    as_type=_observation_type(event.node),
                    input={"summary": REDACTED if redacting() else event.summary},
                    output=_redact_detail(event.detail) if redacting() else event.detail,
                    # Only when there is one. Attaching `model: None` to every
                    # deterministic node invents a field the run does not have.
                    metadata=(
                        {"model": event.detail["model"]}
                        if isinstance(event.detail, dict) and event.detail.get("model")
                        else None
                    ),
                )
                child.end()

            root.update(output=_summarise_output(state))
            if usage and usage.calls:
                root.update(
                    usage_details={
                        "input": usage.input_tokens,
                        "output": usage.output_tokens,
                    },
                    cost_details={"total": usage.usd},
                )
            if elapsed_ms is not None:
                root.update(metadata={"elapsed_ms": round(elapsed_ms, 1)})
            url = client.get_trace_url()
        client.flush()
        return url
    except Exception:
        # Deliberately silent in production: observability that takes the system
        # down with it is worse than none, and the in-state trace still holds
        # everything this would have shown.
        #
        # But silence has a cost that showed up immediately in testing, where a
        # bug in a test double looked like a wrong assertion rather than a
        # crash. `LANGFUSE_DEBUG=1` re-raises, so the guarantee stays intact for
        # users while remaining diagnosable by whoever is working on it.
        if optional("LANGFUSE_DEBUG"):
            raise
        return None


@contextmanager
def timed():
    """Wall-clock for a run, since the state trace records what and not when."""
    started = time.perf_counter()
    box: dict[str, float] = {}
    try:
        yield box
    finally:
        box["elapsed_ms"] = (time.perf_counter() - started) * 1000


def run_traced(
    graph,
    state: AgentState,
    usage: Usage | None = None,
    session_id: str | None = None,
) -> tuple[AgentState, str | None]:
    """Invoke the agent and mirror the run into Langfuse. Returns (state, url).

    **This is the request boundary, and tracing belongs here rather than inside
    `build_agent`.** Three reasons, in increasing order of importance:

    1. `build_agent` returns a compiled graph. Export happens after `invoke`
       returns, so wrapping it would mean proxying the graph's interface to bolt
       on a concern it does not have.
    2. Tracing is a property of a *request*, not of the agent. One request, one
       trace.
    3. **The evals must not trace.** They run 92 scenarios almost entirely from
       cache, so those spans would record decisions that never called a model,
       at eval volume, burying the real traces. Observability nobody trusts is
       worse than none, which is the same reason `export` swallows its own
       failures rather than taking a run down.

    So `eval/` calls `graph.invoke()` directly and stays untraced, while the API
    and demo call this.
    """
    with timed() as clock:
        final = graph.invoke(state)
    url = export(
        final,
        usage=usage,
        elapsed_ms=clock.get("elapsed_ms"),
        session_id=session_id,
    )
    return final, url
