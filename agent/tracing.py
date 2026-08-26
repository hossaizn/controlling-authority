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


def _observation_type(node: str) -> str:
    if node in _GENERATION_NODES:
        return "generation"
    if node in _RETRIEVER_NODES:
        return "retriever"
    return "span"


def _summarise_input(state: AgentState) -> dict[str, Any]:
    context = state.get("employee_context")
    return {
        "question": state.get("question"),
        "as_of": str(state.get("as_of")),
        "employee_context": (
            {k: v for k, v in context.model_dump().items() if v is not None}
            if context
            else {}
        ),
    }


def _summarise_output(state: AgentState) -> dict[str, Any]:
    verification = state.get("verification")
    resolution = state.get("resolution")
    return {
        "route": state.get("route"),
        "controlling_authority": resolution.controlling if resolution else None,
        "precedence_rule": resolution.rule if resolution else None,
        "citations": state.get("citations", []),
        "verified": bool(verification and verification.passed),
        "answer": state.get("answer"),
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
                    input={"summary": event.summary},
                    output=event.detail,
                    metadata={"model": event.detail.get("model")},
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
        # Deliberately silent. Observability that takes the system down with it
        # is worse than no observability, and the in-state trace still holds
        # everything this would have shown.
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
