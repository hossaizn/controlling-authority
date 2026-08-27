"""Tests for the assembly, and specifically for the control arm.

**Found by mutation testing, not by review.** `build_baseline` could be edited to
wire the precedence resolver instead of the naive one and all 523 tests stayed
green. That is the control arm of the project's headline claim: precedence as
code against a system that trusts the top-ranked passage, +22.8 points (DL-23).
A baseline that quietly runs the agent reports a delta of zero and looks like a
refuted thesis rather than a broken harness.

It is also DL-32's defect class a second time: *the demo claimed a comparison
the baseline never makes*. The remedy there was to make the baseline share the
agent's code path. Nothing pinned the one place they are supposed to differ.
"""

from __future__ import annotations

import pytest

import agent.build as build
from agent.build import build_agent, build_baseline, naive_resolve

NODES = ("triage", "clarify", "retrieve", "resolve", "compose", "verify",
         "refuse", "escalate")


@pytest.fixture
def captured(monkeypatch):
    """Capture what gets wired, without compiling a graph or touching a store."""
    seen: dict[str, dict] = {}

    def fake_build_graph(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return "graph"

    monkeypatch.setattr(build, "build_graph", fake_build_graph)
    return seen


def wiring(fn, captured) -> dict[str, str]:
    """Node identity by qualified name. `make_triage` and friends return a fresh
    closure per call, so `is` comparison would report every node as different
    and the test would pass for the wrong reason."""
    fn(store=None)
    return {k: getattr(v, "__qualname__", repr(v)) for k, v in captured.items()}


def test_the_baseline_wires_the_naive_resolver(captured) -> None:
    build_baseline(store=None)
    assert captured["resolve"] is naive_resolve


def test_the_agent_does_not_wire_the_naive_resolver(captured) -> None:
    build_agent(store=None)
    assert captured["resolve"] is not naive_resolve
    assert captured["resolve"].__qualname__ == "make_resolve.<locals>.resolve"


def test_the_two_graphs_differ_in_exactly_one_node(captured) -> None:
    """The docstring's claim: one code path with one component swapped, so the
    comparison isolates precedence rather than two implementations that drifted.
    Nothing tested it."""
    agent_wiring = wiring(build_agent, captured)
    baseline_wiring = wiring(build_baseline, captured)

    assert set(agent_wiring) == set(NODES)
    differing = [k for k in NODES if agent_wiring[k] != baseline_wiring[k]]
    assert differing == ["resolve"], (
        f"baseline should differ from the agent only at resolve, got {differing}"
    )


def test_the_baseline_keeps_verify(captured) -> None:
    """Stripping it too would make the baseline lose on groundedness as well as
    precedence, and the comparison is supposed to isolate one thing."""
    build_baseline(store=None)
    assert captured["verify"].__qualname__ == "make_verify.<locals>.verify"
