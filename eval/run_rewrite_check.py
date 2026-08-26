"""Does triage's rewritten query retrieve worse than the raw question?

The question the regression gate was built to answer, and the reason it was built
before the agent existed. An end-to-end score cannot answer it: better routing
offsets worse retrieval and the total looks fine while the system got worse at
the thing it was already good at.

Same 57 scoreable scenarios, same collection, same filters, same code path. The
only variable is the query text.

    uv run python -m eval.run_rewrite_check

Note this deliberately ignores what triage ROUTED each scenario to. A mis-route
is a routing failure, already counted in `eval/run_routes.py`, and folding it in
here would mix two effects into one number.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent.models import StructuredCaller, Usage
from agent.nodes.triage import PROMPT_VERSION, make_triage
from agent.state import initial_state
from eval.metrics.retrieval import aggregate, aggregate_by_slice
from eval.regression import check, format_report
from eval.run_retrieval import RUNS_DIR, run_scenarios, scoreable
from eval.scenarios.schema import Scenario
from retrieval.embed import get_provider
from retrieval.store import ChunkStore

ADOPTED_MODEL = "voyage-law"
ADOPTED_STRATEGY = "structure"


def rewritten_queries(scenarios: list[Scenario]) -> tuple[dict[str, str], Usage]:
    usage = Usage()
    triage = make_triage(StructuredCaller(usage=usage))
    queries = {}
    for s in scenarios:
        out = triage(initial_state(s.question, s.employee_context, s.as_of_date))
        queries[s.scenario_id] = out["rewritten_query"]
    return queries, usage


def main() -> int:
    scenarios = scoreable()
    queries, usage = rewritten_queries(scenarios)
    print(f"triage: {usage.summary()}")

    store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)
    # rebuild=False: the collection is already indexed and re-embedding the
    # corpus to change the query would be paying to hold a constant constant.
    scores = run_scenarios(
        store, scenarios, query_of=lambda s: queries[s.scenario_id]
    )

    by_slice = aggregate_by_slice(scores)
    observed = {name: s["recall@10"] for name, s in by_slice.items()}

    verdicts = check(observed)
    print()
    print(format_report(verdicts))
    print()
    print(f"overall recall@10 with rewritten queries: {aggregate(scores)['recall@10']:.4f}")

    regressions = [v for v in verdicts if not v.passed and v.name != "conflict"]
    print()
    if regressions:
        print(f"GATE FAILS: {len(regressions)} slice(s) regressed on the rewrite.")
    else:
        print("No slice regressed on the rewrite.")

    path = Path(RUNS_DIR) / f"rewrite_check_{PROMPT_VERSION}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "by_slice": by_slice,
                "overall": aggregate(scores),
                "queries": queries,
                "verdicts": [
                    {
                        "slice": v.name,
                        "baseline": v.baseline,
                        "observed": v.observed,
                        "passed": v.passed,
                        "reason": v.reason,
                    }
                    for v in verdicts
                ],
            },
            indent=2,
        )
    )
    print(f"saved {path}")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
