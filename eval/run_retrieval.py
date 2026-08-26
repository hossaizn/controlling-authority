"""Retrieval evaluation: run the scenario set against a configured store.

**These are oracle-filter numbers.** Jurisdiction and as-of date are taken from
each scenario's metadata rather than derived from its question, because query
rewriting is not built (DL-16). So this measures retrieval *given perfect query
understanding*, which is the right way to isolate one variable and is not the
same as end-to-end performance. Reporting it as though it were would flatter the
system by exactly the amount the unbuilt component would cost.

Only scenarios carrying `required_citations` are scored. Clarify, refuse and
escalate have no retrieval target: whether the agent should have asked a question
rather than answered is a Phase 6 measurement.

Verified and unverified scenarios are reported **separately and never pooled**.
Under DL-3 a citation drafted from recall is not ground truth, and averaging the
two would produce one number that is part measurement and part guess.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from eval.metrics.retrieval import RetrievalScore, aggregate, aggregate_by_slice, score_one
from eval.scenarios.loader import load_all
from eval.scenarios.schema import Scenario
from ingest.corpus import build_corpus
from retrieval.chunking import chunk_corpus
from retrieval.embed import EmbeddingProvider
from retrieval.store import ChunkStore

RUNS_DIR = Path(__file__).resolve().parent / "runs"
CORPUS_SNAPSHOT = date(2026, 8, 1)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def scoreable(scenarios: list[Scenario] | None = None) -> list[Scenario]:
    return [s for s in (scenarios or load_all()) if s.required_citations]


def run_scenarios(
    store: ChunkStore, scenarios: list[Scenario], limit: int = 10
) -> list[RetrievalScore]:
    # One request for every question, rather than one per search. Providers
    # rate-limit on requests as well as tokens, and 57 individual calls at three
    # per minute is twenty minutes of waiting for work that fits in three.
    embed_many = getattr(store.provider, "embed_queries", None)
    vectors = (
        embed_many([s.question for s in scenarios])
        if embed_many
        else [None] * len(scenarios)
    )

    scores: list[RetrievalScore] = []
    for scenario, vector in zip(scenarios, vectors, strict=True):
        hits = store.search(
            scenario.question,
            # Oracle filters. See the module docstring.
            jurisdiction=scenario.employee_context.state,
            as_of=scenario.as_of_date,
            limit=limit,
            query_vector=vector,
        )
        scores.append(
            score_one(
                scenario_id=scenario.scenario_id,
                slice_name=scenario.slice,
                retrieved_citations=[h.citation for h in hits],
                required_citations=scenario.required_citations,
                forbidden_citations=scenario.forbidden_citations,
            )
        )
    return scores


def evaluate(
    provider: EmbeddingProvider,
    strategy: str,
    rebuild: bool = True,
    limit: int = 10,
) -> dict:
    """Index the corpus under one configuration and score the scenario set."""
    store = ChunkStore(provider, strategy=strategy)
    if rebuild:
        store.recreate()
        store.index(chunk_corpus(build_corpus(observed_on=CORPUS_SNAPSHOT), strategy=strategy))

    all_scoreable = scoreable()
    verified_ids = {s.scenario_id for s in all_scoreable if s.verified}

    scores = run_scenarios(store, all_scoreable, limit=limit)
    verified = [s for s in scores if s.scenario_id in verified_ids]
    unverified = [s for s in scores if s.scenario_id not in verified_ids]

    return {
        "config": {
            "provider": provider.spec.provider,
            "model": provider.spec.model,
            "dimensions": provider.spec.dimensions,
            "strategy": strategy,
            "limit": limit,
            "corpus_snapshot": CORPUS_SNAPSHOT.isoformat(),
        },
        "git_sha": git_sha(),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        # Kept apart on purpose. Pooling them would average a measurement with a
        # guess and report the result as one number.
        "verified": {
            "overall": aggregate(verified),
            "by_slice": aggregate_by_slice(verified),
        },
        "unverified_indicative": {
            "overall": aggregate(unverified),
            "by_slice": aggregate_by_slice(unverified),
        },
        "scores": [asdict(s) for s in scores],
    }


def save(report: dict, name: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
