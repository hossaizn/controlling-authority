"""The protection layer: what stands between a public demo and an unbounded bill.

**Ordered by what each check protects, cheapest and most certain first.**

1. **Input length.** Free, deterministic, and the only one that bounds the size
   of a single request rather than their number.
2. **The global circuit breaker.** Checked before the per-caller limits because
   it is the one that bounds *spend*. Per-IP and per-session limits are fairness
   mechanisms; only this one is a budget. If it is reached, nothing gets through
   regardless of who is asking.
3. **Per-IP**, then **per-session**. Fairness between callers.

**Pre-computed answers bypass all of it, deliberately.** They cost nothing, so
rate-limiting them would only degrade the path most reviewers take while
protecting a budget they never touch. See `api/precomputed.py`.

**In-memory, single instance, and that is a real limitation.** The spec ships one
Fly.io instance, so per-process counters are correct there and wrong the moment a
second replica exists: each would enforce its own budget and the global cap would
become N times what it says. Written down rather than discovered, and the fix is
a shared store, not a bigger number.

**The clock is injected.** A rate limiter tested against the real clock either
sleeps or asserts almost nothing, and DL-10's rule is to assert values.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ingest.settings import optional

HOUR_SECONDS = 3600.0
DAY_SECONDS = 86400.0


def _limit(name: str, default: str) -> int:
    return int(optional(name, default))


@dataclass(frozen=True)
class Verdict:
    """Why a request was refused, in terms the caller can act on.

    `reason` is written for the person who hit it. A limiter that says only
    "429" teaches nobody anything, and the demo's whole point is being legible.
    """

    allowed: bool
    reason: str = ""
    retry_after_seconds: int | None = None
    limit_hit: str | None = None


ALLOWED = Verdict(allowed=True)


class SlidingWindow:
    """Counts events in a trailing window, per key.

    A fixed calendar window lets a caller spend the whole budget in the last
    second of one window and the whole budget again in the first second of the
    next, which is twice the intended rate at the moment it matters most.
    """

    def __init__(self, span_seconds: float, now: Callable[[], float] = time.monotonic):
        self.span = span_seconds
        self._now = now
        self._events: dict[str, deque[float]] = {}

    #: Cap on distinct keys held at once. Session ids and forwarded IPs are both
    #: client-supplied and unauthenticated, so a caller can mint unlimited keys.
    #: A review drove 500 refused requests with rotating values and every one
    #: allocated two permanent deques. Eviction bounds memory; it cannot be
    #: bypassed for budget because the GLOBAL window uses a single fixed key
    #: ("all") that no eviction can reach.
    MAX_KEYS = 20_000

    def _prune(self, key: str, now: float) -> deque[float]:
        events = self._events.setdefault(key, deque())
        while events and now - events[0] > self.span:
            events.popleft()
        if len(self._events) > self.MAX_KEYS:
            self._evict(now, keep=key)
        return events

    def _evict(self, now: float, keep: str) -> None:
        """Drop keys whose windows are empty, oldest first.

        Only expired keys are dropped, so eviction never forgives a caller who
        is still inside their window. If every key is live the map is left
        alone: growing memory is preferable to handing out free budget.
        """
        for key in list(self._events):
            if key == keep or key == "all":
                continue
            events = self._events[key]
            while events and now - events[0] > self.span:
                events.popleft()
            if not events:
                del self._events[key]

    def count(self, key: str) -> int:
        return len(self._prune(key, self._now()))

    def record(self, key: str) -> None:
        now = self._now()
        self._prune(key, now).append(now)

    def retry_after(self, key: str) -> int:
        """Seconds until the oldest event leaves the window."""
        events = self._prune(key, self._now())
        if not events:
            return 0
        return max(1, int(self.span - (self._now() - events[0])) + 1)

    def forget(self, key: str) -> None:
        self._events.pop(key, None)


@dataclass
class Protection:
    """Every limit in one object so the order of checks is visible in one place."""

    max_input_chars: int = field(default_factory=lambda: _limit("MAX_INPUT_CHARS", "1000"))
    per_ip_hourly: int = field(default_factory=lambda: _limit("PER_IP_HOURLY_LIMIT", "20"))
    per_session_daily: int = field(
        default_factory=lambda: _limit("PER_SESSION_DAILY_LIMIT", "40")
    )
    global_daily: int = field(
        default_factory=lambda: _limit("GLOBAL_DAILY_REQUEST_LIMIT", "500")
    )
    now: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._ip = SlidingWindow(HOUR_SECONDS, self.now)
        self._session = SlidingWindow(DAY_SECONDS, self.now)
        self._global = SlidingWindow(DAY_SECONDS, self.now)

    # -- inspection, for the health endpoint and the demo footer -------------

    def remaining_global(self) -> int:
        return max(0, self.global_daily - self._global.count("all"))

    def snapshot(self) -> dict[str, int]:
        return {
            "global_daily_limit": self.global_daily,
            "global_remaining": self.remaining_global(),
            "per_ip_hourly_limit": self.per_ip_hourly,
            "per_session_daily_limit": self.per_session_daily,
            "max_input_chars": self.max_input_chars,
        }

    # -- the gate ------------------------------------------------------------

    def check(self, ip: str, session_id: str, question: str) -> Verdict:
        """Decide without recording. `record` is called only if the work runs.

        Split so a request refused downstream, or served from the pre-computed
        set, does not consume budget it never spent. Counting on check would make
        the global cap drift down against work that never happened, which is the
        same error as billing an all-cache eval run.
        """
        if len(question) > self.max_input_chars:
            return Verdict(
                allowed=False,
                limit_hit="input_length",
                reason=(
                    f"That question is {len(question)} characters and the limit is "
                    f"{self.max_input_chars}. Please shorten it."
                ),
            )

        # Before the per-caller limits: this is the only one that bounds spend.
        if self._global.count("all") >= self.global_daily:
            return Verdict(
                allowed=False,
                limit_hit="global_daily",
                retry_after_seconds=self._global.retry_after("all"),
                reason=(
                    "This demo has reached its daily budget, which exists so a "
                    "public endpoint cannot run up an unbounded bill. The "
                    "pre-computed example scenarios still work and always will, "
                    "because they cost nothing to serve."
                ),
            )

        if self._ip.count(ip) >= self.per_ip_hourly:
            return Verdict(
                allowed=False,
                limit_hit="per_ip_hourly",
                retry_after_seconds=self._ip.retry_after(ip),
                reason=(
                    f"You have used {self.per_ip_hourly} free-text questions this "
                    "hour. The example scenarios are unlimited."
                ),
            )

        if self._session.count(session_id) >= self.per_session_daily:
            return Verdict(
                allowed=False,
                limit_hit="per_session_daily",
                retry_after_seconds=self._session.retry_after(session_id),
                reason=(
                    f"This session has used its {self.per_session_daily} questions "
                    "for today. The example scenarios are unlimited."
                ),
            )

        return ALLOWED

    def record(self, ip: str, session_id: str) -> None:
        """Charge one unit of budget. Called only after work actually ran."""
        self._ip.record(ip)
        self._session.record(session_id)
        self._global.record("all")
