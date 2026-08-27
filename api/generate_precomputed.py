"""Capture the curated scenarios from real runs.

    uv run python -m api.generate_precomputed

**Recorded, never written by hand.** A hand-written answer is a brochure; a
captured run is evidence, and it carries a trace whose precedence rule and
citations were actually derived. It also cannot drift into fiction, because
regenerating it requires the system to still work.

Every question comes from the golden scenario set rather than being invented for
the demo, so the six buttons a reviewer clicks are six cases that are also
measured in `eval/`. A demo showing behaviour the eval never scores is a demo
that can be quietly wrong.

Re-running is nearly free: the model decisions are cached by prompt and question,
so this replays from disk unless a prompt changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.build import build_agent, naive_resolve
from agent.models import StructuredCaller, Usage
from agent.nodes.retrieve import make_retrieve
from agent.nodes.triage import make_triage
from agent.state import initial_state
from api import precomputed
from eval.run_precedence import ADOPTED_MODEL, ADOPTED_STRATEGY
from eval.run_retrieval import CORPUS_SNAPSHOT
from eval.scenarios.loader import load_all
from retrieval.embed import get_provider
from retrieval.store import ChunkStore


def main() -> int:
    scenarios = {s.scenario_id: s for s in load_all()}
    missing = [
        sid for sid in precomputed.CURATED.values() if sid not in scenarios
    ]
    if missing:
        # Loud rather than skipped: a curated key pointing at a scenario that no
        # longer exists means the demo and the eval have diverged.
        print(f"ERROR: curated scenarios not in the golden set: {missing}")
        return 1

    # Written in the same pass as the records, so the scores cannot describe a
    # different run from the answers they sit beside. Loud if absent: a demo
    # claiming honesty via numbers it does not have is worse than one that says
    # it needs regenerating.
    run = Path("eval/runs/end_to_end.json")
    if not run.exists():
        print(
            "ERROR: eval/runs/end_to_end.json is missing. Run "
            "`uv run python -m eval.run_end_to_end` first: the demo quotes its "
            "per-slice scores and must not ship without them."
        )
        return 1
    scored = json.loads(run.read_text())
    precomputed.SCORES.parent.mkdir(parents=True, exist_ok=True)
    precomputed.SCORES.write_text(
        json.dumps(
            {
                "generated_from": str(run),
                "run_at": scored.get("run_at"),
                "model": scored.get("model"),
                "overall": {
                    "fully_correct": round(scored["fully_correct"], 4),
                    "route_accuracy_macro": round(scored["route_accuracy_macro"], 4),
                    "precedence_correct": round(scored["precedence_correct"], 4),
                    "n": scored["n"],
                },
                "by_slice": {
                    k: {
                        "n": v["n"],
                        "route_accuracy": round(v["route"], 3),
                        "fully_correct": round(v["fully_correct"], 3),
                    }
                    for k, v in scored["by_slice"].items()
                },
            },
            indent=2,
        )
    )

    usage = Usage()
    caller = StructuredCaller(usage=usage)
    store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)
    agent = build_agent(store, caller)
    # The baseline's own triage and retrieval are identical to the agent's, so
    # the only variable in the comparison is how the authority is chosen.
    triage = make_triage(caller)
    retrieve = make_retrieve(store)
    provenance = precomputed.current_provenance(CORPUS_SNAPSHOT)

    for key, scenario_id in precomputed.CURATED.items():
        s = scenarios[scenario_id]
        state = initial_state(s.question, s.employee_context, s.as_of_date)
        final = agent.invoke(state)
        expected = {
            "route": s.expected_route,
            "authority": s.expected_authority,
            "acceptable_authorities": list(s.acceptable_authorities),
            "required_citations": list(s.required_citations),
            "slice": s.slice,
        }
        precomputed.save(key, final, provenance, expected)

        # **The baseline arm records its RESOLUTION only, and never composes.**
        #
        # The spec's argument is that the baseline picks the wrong authority and
        # the agent catches that a statute overrides it. That delta lives
        # entirely in `resolve`, which for the baseline is `naive_resolve`: take
        # the top-ranked passage. It needs no model, so the comparison the demo
        # is built around costs nothing and cannot be blocked by an unfunded key.
        #
        # Composing a baseline answer would add two model calls per scenario to
        # restate a conclusion the resolution already shows, and it is exactly
        # what could not be generated when both providers were exhausted.
        naive_state = initial_state(s.question, s.employee_context, s.as_of_date)
        naive_state.update(triage(naive_state))
        naive_state.update(retrieve(naive_state))
        naive_state.update(naive_resolve(naive_state))
        precomputed.save_baseline(key, naive_state, provenance, expected)

        got = final.get("route")
        mark = "ok " if got == s.expected_route else "MISS"
        agent_auth = (final.get("resolution").controlling if final.get("resolution") else None)
        naive_auth = naive_state["resolution"].controlling
        delta = " <- baseline differs" if agent_auth != naive_auth else ""
        print(
            f"  {mark} {key:16} {scenario_id:20} expected={s.expected_route:9} "
            f"got={got:9} authority: agent={agent_auth} naive={naive_auth}{delta}"
        )

    print()
    print(f"cost: {usage.summary()}")
    print(
        "Any MISS above is shown in the demo as-is, with the expectation beside "
        "the result. Swapping in scenarios the agent happens to get right would "
        "make the demo a brochure."
    )
    print(f"provenance: {provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
