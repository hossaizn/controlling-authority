"""Retrieval metrics.

Scored at the level of **citations**, not chunks. A scenario's ground truth
names the sources a correct answer must rest on, and how many chunks a document
happens to split into is an artefact of the strategy under test. Counting chunks
would reward whichever strategy fragments the right document most.

`recall@3` alongside `recall@10` is not decoration: the gap between them is what
DL-16 uses to decide whether reranking is worth building, because it isolates
passages the system found and then buried from passages it never found at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScore:
    scenario_id: str
    slice: str
    recall_at_3: float
    recall_at_10: float
    mrr: float
    forbidden_hit: bool
    required: int
    found_at_10: int


def _first_rank(retrieved: list[str], required: set[str]) -> int | None:
    for position, citation in enumerate(retrieved, start=1):
        if citation in required:
            return position
    return None


def score_one(
    scenario_id: str,
    slice_name: str,
    retrieved_citations: list[str],
    required_citations: list[str],
    forbidden_citations: list[str],
) -> RetrievalScore:
    required = set(required_citations)
    forbidden = set(forbidden_citations)

    def recall(k: int) -> float:
        if not required:
            return 0.0
        seen = set(retrieved_citations[:k])
        return len(required & seen) / len(required)

    rank = _first_rank(retrieved_citations, required)
    return RetrievalScore(
        scenario_id=scenario_id,
        slice=slice_name,
        recall_at_3=recall(3),
        recall_at_10=recall(10),
        mrr=1.0 / rank if rank else 0.0,
        # Superseded provisions and other jurisdictions' law. Their presence is
        # a failure in itself, not a ranking inconvenience: an answer built on
        # one is wrong however well the rest scored.
        forbidden_hit=bool(forbidden & set(retrieved_citations[:10])),
        required=len(required),
        found_at_10=len(required & set(retrieved_citations[:10])),
    )


def aggregate(scores: list[RetrievalScore]) -> dict[str, float]:
    if not scores:
        return {"n": 0}
    n = len(scores)
    return {
        "n": n,
        "recall@3": sum(s.recall_at_3 for s in scores) / n,
        "recall@10": sum(s.recall_at_10 for s in scores) / n,
        "mrr": sum(s.mrr for s in scores) / n,
        "forbidden_rate": sum(s.forbidden_hit for s in scores) / n,
        # DL-16's decision variable. Headroom a reranker could recover, as
        # distinct from documents retrieval never surfaced.
        "rerank_headroom": (
            sum(s.recall_at_10 for s in scores) / n - sum(s.recall_at_3 for s in scores) / n
        ),
    }


def aggregate_by_slice(scores: list[RetrievalScore]) -> dict[str, dict[str, float]]:
    slices: dict[str, list[RetrievalScore]] = {}
    for score in scores:
        slices.setdefault(score.slice, []).append(score)
    return {name: aggregate(group) for name, group in sorted(slices.items())}
