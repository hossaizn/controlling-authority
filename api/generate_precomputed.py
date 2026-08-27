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

from agent.build import build_agent
from agent.models import StructuredCaller, Usage
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

    usage = Usage()
    caller = StructuredCaller(usage=usage)
    store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)
    agent = build_agent(store, caller)
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
        got = final.get("route")
        mark = "ok " if got == s.expected_route else "MISS"
        print(
            f"  {mark} {key:16} {scenario_id:20} "
            f"expected={s.expected_route:9} got={got}"
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
