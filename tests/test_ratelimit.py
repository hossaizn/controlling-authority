"""Rate budget tests. No sleeping in the suite: the clock is injected."""

from __future__ import annotations

import pytest

from retrieval import ratelimit
from retrieval.ratelimit import WINDOW_SECONDS, RateBudget, estimate_tokens


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


def test_an_oversized_request_still_paces_the_next_one(monkeypatch) -> None:
    """Having spent the budget, the following call must WAIT.

    The first version asserted `len(_events) == 1`, which a budget that never
    paced would also satisfy: it tested that the event was recorded, not that
    recording it had any effect.

    The clock is stubbed on BOTH `sleep` and `monotonic`. Stubbing only `sleep`
    made the loop busy-wait for the real sixty seconds, turning a 3-second suite
    into a 78-second one, because time never advanced so the condition never
    cleared. Half a fake clock is worse than none.
    """
    now = {"t": 1000.0}
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(ratelimit.time, "sleep", fake_sleep)
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now["t"])

    budget = RateBudget(max_tokens_per_minute=1_000, max_requests_per_minute=60)
    budget.acquire(5_000)
    assert slept == [], "an oversized request on an empty window proceeds at once"

    budget.acquire(100)
    assert slept, "the call after an oversized one must wait for the window"
    assert sum(slept) >= WINDOW_SECONDS


def test_a_budget_that_can_never_admit_a_request_is_rejected() -> None:
    """Fewer than one request per window can never satisfy `request_ok`, so
    `acquire` spun forever. A silent hang is the worst failure a pacer can have:
    the run looks alive and never progresses, which is what the 429 backoff
    already looked like."""
    with pytest.raises(ValueError, match="at least 1"):
        RateBudget(max_tokens_per_minute=8_000, max_requests_per_minute=0)
