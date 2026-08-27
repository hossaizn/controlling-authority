"""Tests for the protection layer.

The clock is injected throughout. A rate limiter tested against the real clock
either sleeps or asserts almost nothing, and this repo's own rule (DL-10) is to
assert values rather than relationships.
"""

from __future__ import annotations

import pytest

from api.limits import DAY_SECONDS, HOUR_SECONDS, Protection, SlidingWindow


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def guard(clock: Clock, **kw) -> Protection:
    defaults = dict(
        max_input_chars=100,
        per_ip_hourly=3,
        per_session_daily=5,
        global_daily=8,
        now=clock,
    )
    return Protection(**{**defaults, **kw})


# --- the window -------------------------------------------------------------


def test_events_leave_the_window_after_its_span() -> None:
    clock = Clock()
    window = SlidingWindow(60.0, clock)
    window.record("a")
    assert window.count("a") == 1
    clock.advance(61)
    assert window.count("a") == 0


def test_the_window_slides_rather_than_resetting_on_a_boundary() -> None:
    """A fixed calendar window lets a caller spend the whole budget in the last
    second of one window and again in the first second of the next, which is
    twice the intended rate at the moment it matters most."""
    clock = Clock()
    window = SlidingWindow(60.0, clock)
    for _ in range(3):
        window.record("a")
    clock.advance(59)
    assert window.count("a") == 3, "still inside the window"
    clock.advance(2)
    assert window.count("a") == 0


def test_keys_do_not_share_a_budget() -> None:
    clock = Clock()
    window = SlidingWindow(60.0, clock)
    window.record("a")
    assert window.count("b") == 0


def test_retry_after_counts_down_as_the_window_drains() -> None:
    clock = Clock()
    window = SlidingWindow(60.0, clock)
    window.record("a")
    assert window.retry_after("a") == 61
    clock.advance(30)
    assert window.retry_after("a") == 31


def test_retry_after_is_zero_on_an_empty_window() -> None:
    assert SlidingWindow(60.0, Clock()).retry_after("a") == 0


# --- ordering: what each check protects -------------------------------------


def test_input_length_is_rejected_before_any_budget_is_consulted() -> None:
    """Free and deterministic, so it runs first. It also bounds the size of a
    single request, which none of the others do."""
    clock = Clock()
    p = guard(clock, global_daily=0)
    verdict = p.check("1.1.1.1", "s", "x" * 101)
    assert verdict.limit_hit == "input_length"


def test_the_global_breaker_is_checked_before_the_per_caller_limits() -> None:
    """Per-IP and per-session are fairness mechanisms; only the global cap is a
    budget. A fresh caller must not get through once the budget is gone."""
    clock = Clock()
    p = guard(clock, global_daily=2, per_ip_hourly=99)
    for _ in range(2):
        p.record("1.1.1.1", "s1")
    verdict = p.check("9.9.9.9", "brand-new-session", "hello")
    assert verdict.limit_hit == "global_daily"


def test_the_per_ip_limit_fires_before_the_per_session_one() -> None:
    clock = Clock()
    p = guard(clock, per_ip_hourly=2, per_session_daily=99)
    for _ in range(2):
        p.record("1.1.1.1", "s1")
    assert p.check("1.1.1.1", "s1", "hello").limit_hit == "per_ip_hourly"


def test_a_second_session_from_one_ip_still_hits_the_ip_limit() -> None:
    """Otherwise rotating the session id is a free reset."""
    clock = Clock()
    p = guard(clock, per_ip_hourly=2)
    for i in range(2):
        p.record("1.1.1.1", f"s{i}")
    assert p.check("1.1.1.1", "s-fresh", "hello").limit_hit == "per_ip_hourly"


def test_a_session_reaching_its_daily_cap_is_refused_from_a_new_ip() -> None:
    """And rotating the IP must not reset the session either."""
    clock = Clock()
    p = guard(clock, per_ip_hourly=99, per_session_daily=2)
    for i in range(2):
        p.record(f"1.1.1.{i}", "s1")
    assert p.check("5.5.5.5", "s1", "hello").limit_hit == "per_session_daily"


# --- budget is charged for work that ran, not work that was asked for -------


def test_checking_does_not_consume_budget() -> None:
    """Split so a request refused downstream, or served from the pre-computed
    set, does not spend budget it never used."""
    clock = Clock()
    p = guard(clock, global_daily=2)
    for _ in range(10):
        assert p.check("1.1.1.1", "s", "hello").allowed
    assert p.remaining_global() == 2


def test_recording_consumes_exactly_one_unit_across_all_three_windows() -> None:
    clock = Clock()
    p = guard(clock)
    p.record("1.1.1.1", "s")
    assert p.remaining_global() == 7
    assert p._ip.count("1.1.1.1") == 1
    assert p._session.count("s") == 1


# --- recovery ---------------------------------------------------------------


def test_the_ip_limit_clears_after_an_hour() -> None:
    clock = Clock()
    p = guard(clock, per_ip_hourly=1)
    p.record("1.1.1.1", "s")
    assert not p.check("1.1.1.1", "s", "hello").allowed
    clock.advance(HOUR_SECONDS + 1)
    assert p.check("1.1.1.1", "s", "hello").allowed


def test_the_global_breaker_clears_after_a_day() -> None:
    clock = Clock()
    p = guard(clock, global_daily=1)
    p.record("1.1.1.1", "s")
    assert not p.check("2.2.2.2", "other", "hello").allowed
    clock.advance(DAY_SECONDS + 1)
    assert p.check("2.2.2.2", "other", "hello").allowed


def test_an_hour_of_recovery_does_not_clear_the_daily_session_cap() -> None:
    """The windows are different spans on purpose; sharing one would make the
    daily cap an hourly one."""
    clock = Clock()
    p = guard(clock, per_ip_hourly=99, per_session_daily=1)
    p.record("1.1.1.1", "s")
    clock.advance(HOUR_SECONDS + 1)
    assert not p.check("1.1.1.1", "s", "hello").allowed


# --- the refusal has to teach the caller something --------------------------


@pytest.mark.parametrize(
    "limit,kwargs",
    [("global_daily", {"global_daily": 0}), ("per_ip_hourly", {"per_ip_hourly": 0})],
)
def test_a_refusal_names_the_limit_and_when_to_retry(limit, kwargs) -> None:
    clock = Clock()
    p = guard(clock, **kwargs)
    p.record("1.1.1.1", "s")
    verdict = p.check("1.1.1.1", "s", "hello")
    assert verdict.limit_hit == limit
    assert verdict.reason
    assert verdict.retry_after_seconds and verdict.retry_after_seconds > 0


def test_the_budget_refusal_points_at_the_free_path() -> None:
    """A demo that only says "429" teaches nobody anything, and the curated
    scenarios genuinely still work."""
    clock = Clock()
    p = guard(clock, global_daily=0)
    assert "cost nothing" in p.check("1.1.1.1", "s", "hello").reason


def test_the_snapshot_reports_remaining_budget() -> None:
    clock = Clock()
    p = guard(clock, global_daily=3)
    p.record("1.1.1.1", "s")
    snap = p.snapshot()
    assert snap["global_remaining"] == 2
    assert snap["global_daily_limit"] == 3


def test_remaining_never_goes_negative() -> None:
    clock = Clock()
    p = guard(clock, global_daily=1)
    for _ in range(5):
        p.record("1.1.1.1", "s")
    assert p.remaining_global() == 0
