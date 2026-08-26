"""Metric tests. Pinned values, not relationships."""

from __future__ import annotations

from eval.metrics.retrieval import aggregate, aggregate_by_slice, score_one


def s(retrieved, required, forbidden=(), sid="x", sl="conflict"):
    return score_one(sid, sl, list(retrieved), list(required), list(forbidden))


def test_perfect_retrieval_scores_one() -> None:
    score = s(["A", "B"], ["A"])
    assert score.recall_at_3 == 1.0 and score.recall_at_10 == 1.0
    assert score.mrr == 1.0


def test_recall_is_the_fraction_of_required_citations_found() -> None:
    score = s(["A", "X", "Y"], ["A", "B"])
    assert score.recall_at_10 == 0.5
    assert score.found_at_10 == 1 and score.required == 2


def test_recall_at_3_and_10_diverge_when_the_answer_is_buried() -> None:
    """The gap between them is DL-16's decision variable for reranking."""
    retrieved = ["X", "Y", "Z", "Q", "A"]
    score = s(retrieved, ["A"])
    assert score.recall_at_3 == 0.0
    assert score.recall_at_10 == 1.0
    assert score.mrr == 0.2  # first correct hit at rank 5


def test_headroom_isolates_buried_answers_from_missing_ones() -> None:
    buried = [s(["X", "Y", "Z", "A"], ["A"], sid="buried")]
    missing = [s(["X", "Y", "Z"], ["A"], sid="missing")]
    assert aggregate(buried)["rerank_headroom"] == 1.0
    # Never retrieved: a reranker cannot help, so headroom must be zero.
    assert aggregate(missing)["rerank_headroom"] == 0.0
    assert aggregate(missing)["recall@10"] == 0.0


def test_forbidden_citations_are_a_failure_not_a_ranking_issue() -> None:
    score = s(["LEAVE-004-v2", "LEAVE-004-v1"], ["LEAVE-004-v2"], ["LEAVE-004-v1"])
    assert score.recall_at_10 == 1.0
    assert score.forbidden_hit is True


def test_forbidden_beyond_rank_10_is_not_counted() -> None:
    retrieved = [f"pad{i}" for i in range(10)] + ["BAD"]
    assert s(retrieved, ["pad0"], ["BAD"]).forbidden_hit is False


def test_aggregate_reports_n_and_averages() -> None:
    scores = [s(["A"], ["A"], sid="1"), s(["X"], ["A"], sid="2")]
    result = aggregate(scores)
    assert result["n"] == 2
    assert result["recall@10"] == 0.5
    assert result["mrr"] == 0.5


def test_aggregate_of_nothing_is_not_a_crash_or_a_fake_score() -> None:
    assert aggregate([]) == {"n": 0}


def test_by_slice_keeps_slices_separate() -> None:
    scores = [
        s(["A"], ["A"], sid="1", sl="conflict"),
        s(["X"], ["A"], sid="2", sl="straightforward"),
    ]
    grouped = aggregate_by_slice(scores)
    assert grouped["conflict"]["recall@10"] == 1.0
    assert grouped["straightforward"]["recall@10"] == 0.0
