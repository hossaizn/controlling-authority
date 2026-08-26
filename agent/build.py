"""Assemble the agent.

One place where the real nodes are wired into the graph, so the eval, the API and
the demo all run the same object. Two assemblies that were meant to be identical
and drifted is a class of bug no test catches, because each one passes its own.

`build_baseline` is the other half of the argument. It is the SAME graph with the
precedence machinery removed, so the comparison in Phase 9 is one code path with
one component swapped rather than two implementations that might differ for
reasons nobody intended.
"""

from __future__ import annotations

from agent.graph import build_graph
from agent.models import HAIKU, StructuredCaller
from agent.nodes.clarify import clarify
from agent.nodes.compose import make_compose
from agent.nodes.resolve import make_resolve
from agent.nodes.retrieve import make_retrieve
from agent.nodes.terminal import escalate, refuse
from agent.nodes.triage import make_triage
from agent.nodes.verify import VERIFY_MODEL, make_verify
from agent.state import AgentState, LayerFinding, Resolution, TraceEvent
from retrieval.store import ChunkStore


def build_agent(
    store: ChunkStore,
    caller: StructuredCaller | None = None,
    model: str = HAIKU,
    verify_model: str = VERIFY_MODEL,
):
    caller = caller or StructuredCaller()
    return build_graph(
        triage=make_triage(caller, model),
        clarify=clarify,
        retrieve=make_retrieve(store),
        resolve=make_resolve(caller, model),
        compose=make_compose(caller, model),
        verify=make_verify(caller, verify_model),
        refuse=refuse,
        escalate=escalate,
    )


def naive_resolve(state: AgentState) -> dict:
    """What a system that trusts the most relevant document concludes.

    No precedence rules, no layer comparison: the top-ranked passage wins. This
    is not a straw man. It is what a competent RAG pipeline does, and on this
    corpus it is right 63.2% of the time (DL-23), which is high enough to look
    fine and low enough to be wrong on the cases that matter.
    """
    hits = state.get("retrieved", [])
    if not hits:
        return {
            "resolution": Resolution(controlling=None, rule="not_reached"),
            "trace": [TraceEvent(node="resolve", summary="nothing retrieved")],
        }

    top = hits[0]
    resolution = Resolution(
        controlling=top.authority_layer,
        rule="not_reached",  # no rule fired; that is the whole point
        considered=[
            LayerFinding(
                layer=top.authority_layer,
                speaks_to_question=True,
                outcome="grants",
                citation=top.citation,
                says=top.heading,
                generosity_rank=1,
            )
        ],
    )
    return {
        "resolution": resolution,
        "trace": [
            TraceEvent(
                node="resolve",
                summary=f"used the closest matching passage, {top.citation}",
                detail={
                    "controlling": top.authority_layer,
                    "rule": "none, the top-ranked passage was taken as authoritative",
                    "citation": top.citation,
                },
            )
        ],
    }


def build_baseline(
    store: ChunkStore, caller: StructuredCaller | None = None, model: str = HAIKU
):
    """The same graph with precedence removed, for the demo's side-by-side.

    `verify` is left in. Stripping it too would make the baseline lose on
    groundedness as well as precedence, and the comparison is supposed to isolate
    one thing.
    """
    caller = caller or StructuredCaller()
    return build_graph(
        triage=make_triage(caller, model),
        clarify=clarify,
        retrieve=make_retrieve(store),
        resolve=naive_resolve,
        compose=make_compose(caller, model),
        verify=make_verify(caller, model),
        refuse=refuse,
        escalate=escalate,
    )
