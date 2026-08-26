"""The graph. Explicit nodes, because a reviewer can read a state machine and
cannot read a prompt loop.

Nodes are injected rather than imported at module level. That is not a testing
convenience: Phase 9's baseline toggle runs the *same* graph with a reduced set
of nodes so naive RAG and the full agent are compared on one code path rather
than two implementations that might differ for reasons nobody intended. Passing
them in is what makes that honest.

This module is the skeleton. Every node here is a stub that records its own
visit and nothing else, so the wiring can be tested before any model is called.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from agent.state import AgentState, TraceEvent

Node = Callable[[AgentState], dict]

# Nodes that end a run. Each is a legitimate outcome, not a failure: refusing and
# escalating are answers (spec, "Refuse and escalate").
TERMINAL_ROUTES = {"clarify": "clarify", "refuse": "refuse", "escalate": "escalate"}


def _stub(name: str, **updates) -> Node:
    def node(state: AgentState) -> dict:
        return {"trace": [TraceEvent(node=name, summary=f"{name} (stub)")], **updates}

    return node


def _route_after_triage(state: AgentState) -> str:
    """Dispatch on the route triage decided.

    Raises rather than defaulting when the route is missing. A graph that
    silently falls through to `retrieve` when triage failed would answer a
    question it had already determined it could not answer, and nothing
    downstream could tell that apart from a normal run. Same reasoning as DL-8's
    refusal to invent an effective date.
    """
    route = state.get("route")
    if route is None:
        raise ValueError("triage returned no route; refusing to guess one")
    if route in TERMINAL_ROUTES:
        return TERMINAL_ROUTES[route]
    if route == "answer":
        return "retrieve"
    raise ValueError(f"unknown route {route!r}")


def build_graph(
    triage: Node | None = None,
    clarify: Node | None = None,
    retrieve: Node | None = None,
    resolve: Node | None = None,
    compose: Node | None = None,
    verify: Node | None = None,
    refuse: Node | None = None,
    escalate: Node | None = None,
):
    """Wire the graph. Any node left as None gets a stub that only records itself."""
    graph = StateGraph(AgentState)

    # The stub triage routes to `answer` so the default graph exercises the long
    # path. The other branches are reached by injecting a triage that chooses
    # them, which is how the real node will work too.
    graph.add_node("triage", triage or _stub("triage", route="answer"))
    graph.add_node("clarify", clarify or _stub("clarify"))
    graph.add_node("retrieve", retrieve or _stub("retrieve"))
    graph.add_node("resolve", resolve or _stub("resolve"))
    graph.add_node("compose", compose or _stub("compose"))
    graph.add_node("verify", verify or _stub("verify"))
    graph.add_node("refuse", refuse or _stub("refuse"))
    graph.add_node("escalate", escalate or _stub("escalate"))

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        _route_after_triage,
        ["clarify", "retrieve", "refuse", "escalate"],
    )

    # The answering path. Precedence resolution sits between retrieval and
    # composition on purpose: the answer is drafted from the controlling
    # provision, so which one controls has to be settled before anything is
    # written. Composing first and checking afterwards would mean arguing a model
    # out of an answer it had already committed to.
    graph.add_edge("retrieve", "resolve")
    graph.add_edge("resolve", "compose")
    graph.add_edge("compose", "verify")
    graph.add_edge("verify", END)

    for terminal in TERMINAL_ROUTES.values():
        graph.add_edge(terminal, END)

    return graph.compile()
