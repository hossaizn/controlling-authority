"""Rate budget tests. No sleeping in the suite: the clock is injected."""

from __future__ import annotations

from retrieval.ratelimit import RateBudget, estimate_tokens


def test_token_estimate_is_conservative() -> None:
    """Under-counting walks into the ceiling, so the estimate must not be low."""
    text = "x" * 350
    assert estimate_tokens([text]) >= 100


def test_a_request_within_budget_does_not_wait() -> None:
    budget = RateBudget(max_tokens_per_minute=10_000, max_requests_per_minute=10)
    assert budget.acquire(1_000) == 0.0


def test_requests_are_admitted_until_the_token_budget_is_spent() -> None:
    budget = RateBudget(max_tokens_per_minute=10_000, max_requests_per_minute=100)
    for _ in range(10):
        assert budget.acquire(1_000) == 0.0
    # The eleventh would exceed the ceiling and must not be admitted instantly.
    assert budget._events and sum(t for _, t in budget._events) == 10_000


def test_request_count_binds_independently_of_tokens() -> None:
    """Either budget can be the constraint: requests when batches are small,
    tokens when they are large."""
    budget = RateBudget(max_tokens_per_minute=1_000_000, max_requests_per_minute=3)
    for _ in range(3):
        assert budget.acquire(1) == 0.0
    assert len(budget._events) == 3


def test_the_window_prunes_old_events() -> None:
    budget = RateBudget(max_tokens_per_minute=10_000, max_requests_per_minute=100)
    budget.acquire(9_000)
    # Age the recorded event past the window rather than sleeping through it.
    stamp, tokens = budget._events[0]
    budget._events[0] = (stamp - 61.0, tokens)
    assert budget.acquire(9_000) == 0.0


def test_a_request_larger_than_the_whole_budget_does_not_crash() -> None:
    """Reachable, and it raised IndexError.

    Groq's free tier deducts the reserved max_tokens rather than the tokens
    produced, so one `resolve` call reserves almost the entire per-minute budget
    by itself. With an empty window there is no oldest event to wait on, and the
    sleep computation indexed into an empty deque.
    """
    budget = RateBudget(max_tokens_per_minute=1_000, max_requests_per_minute=60)
    assert budget.acquire(5_000) == 0.0


def test_an_oversized_request_still_paces_the_next_one() -> None:
    """Having spent the budget, the following call must wait rather than sail
    through on an empty window."""
    budget = RateBudget(max_tokens_per_minute=1_000, max_requests_per_minute=60)
    budget.acquire(5_000)
    assert len(budget._events) == 1
