"""The agent against the naive baseline, end to end, on every metric.

DL-23 compared the two on precedence alone. That answered "does the precedence
machinery pick the right authority more often", which is the interesting question
but not the one a user asks. This runs both graphs over all 92 scenarios and
compares what actually comes out.

**Same graph, one component swapped.** `build_baseline` replaces `resolve` with
"the top-ranked passage is authoritative" and changes nothing else: same triage,
same retrieval, same compose, same verify. Two separate implementations would
differ for reasons nobody intended and the delta would measure the difference
between two codebases rather than the difference between two ideas.

**Why this number is the one that matters.** `fully_correct` is a five-way
conjunction, so its absolute value is low by construction and means little on its
own. Against a baseline built the same way, the difference means everything.

    uv run python -m eval.run_baseline_compare
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent.build import build_agent, build_baseline
from agent.models import StructuredCaller, Usage
from agent.state import initial_state, nodes_visited
from eval.run_end_to_end import EndToEnd, _rate
from eval.run_precedence import ADOPTED_MODEL, ADOPTED_STRATEGY
from eval.run_routes import score as score_routes
from eval.scenarios.loader import load_all
from retrieval.embed import get_provider
from retrieval.store import ChunkStore

METRICS = [
    ("precedence correct", "precedence_correct", True),
    ("required citations present", "required_present", True),
    ("named the beaten source", "addressed", None),
    ("passed verification", "verified", False),
    ("forbidden citation leaked", "forbidden_present", False),
    ("FULLY CORRECT", "fully_correct", False),
]


def run_graph(graph, scenarios, usage, label) -> list[EndToEnd]:
    results = []
    for i, s in enumerate(scenarios, 1):
        final = graph.invoke(initial_state(s.question, s.employee_context, s.as_of_date))
        results.append(EndToEnd(s, final, nodes_visited(final)))
        if i % 25 == 0 or i == len(scenarios):
            print(f"  {label} {i}/{len(scenarios)}  {usage.summary()}", flush=True)
    return results


def subset(results, attr_scope):
    if attr_scope is True:
        return [r for r in results if r.scenario.expected_route == "answer"]
    if attr_scope is None:
        return [r for r in results if r.scenario.must_address]
    return results


def main() -> int:
    scenarios = load_all()
    usage = Usage()
    caller = StructuredCaller(usage=usage)
    store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)

    agent = run_graph(build_agent(store, caller), scenarios, usage, "agent   ")
    naive = run_graph(build_baseline(store, caller), scenarios, usage, "baseline")

    print()
    print("Same graph, same retrieval, same compose and verify.")
    print("The only difference is whether precedence rules or the top-ranked")
    print("passage decides which authority controls.")
    print()
    print(f"{'metric':30} {'naive':>7} {'agent':>7} {'delta':>9}")
    rows = {}
    for name, attr, scope in METRICS:
        a, b = subset(agent, scope), subset(naive, scope)
        av, bv = _rate(a, attr), _rate(b, attr)
        rows[attr] = {"naive": bv, "agent": av}
        print(f"{name:30} {bv:>7.3f} {av:>7.3f} {100*(av-bv):>+8.1f}p")

    a_routes = score_routes(
        [r.scenario for r in agent], {r.scenario.scenario_id: r.final for r in agent}
    )
    print()
    print(f"{'route accuracy, macro':30} {'same':>7} {a_routes.macro_accuracy:>7.3f}")
    print("(routing is upstream of the swap, so it is identical by construction)")

    print()
    print(f"{'slice':16} {'n':>3} {'naive full':>11} {'agent full':>11} {'delta':>9}")
    slices = sorted({r.scenario.slice for r in agent})
    by_slice = {}
    for name in slices:
        a = [r for r in agent if r.scenario.slice == name]
        b = [r for r in naive if r.scenario.slice == name]
        av, bv = _rate(a, "fully_correct"), _rate(b, "fully_correct")
        by_slice[name] = {"n": len(a), "naive": bv, "agent": av}
        print(f"{name:16} {len(a):>3} {bv:>11.3f} {av:>11.3f} {100*(av-bv):>+8.1f}p")

    print()
    print(f"cost: {usage.summary()}")

    path = Path(__file__).resolve().parent / "runs" / "baseline_compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "n": len(scenarios),
                "metrics": rows,
                "by_slice": by_slice,
                "route_accuracy_macro": a_routes.macro_accuracy,
                "usage": {
                    "calls": usage.calls,
                    "cached": usage.cached,
                    "usd": round(usage.usd, 4),
                },
            },
            indent=2,
        )
    )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
