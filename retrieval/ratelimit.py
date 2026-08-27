"""A token-and-request budget that paces calls instead of colliding with them.

The first attempt at this was reactive: send as fast as possible, catch the rate
limit, back off, retry. That is the wrong shape for the constraint. An unpaid
Voyage account allows 3 requests and 10,000 tokens per minute, so with batches of
roughly 7,000 tokens the sustainable rate is about one request per minute. A
reactive strategy spends its whole retry budget rediscovering that on every call,
and the first run died 264 chunks into 300 having learned nothing it could carry
forward.

Pacing proactively means the limit is never hit at all, so retries become a
safety net for genuine hiccups rather than the primary mechanism.

Both budgets are tracked, because either can bind. Requests bind when batches are
small; tokens bind when they are large.
"""

from __future__ import annotations

import time
from collections import deque

# Headroom under the documented ceilings. The server's window and ours are not
# aligned, so running at exactly the limit still trips it.
DEFAULT_MAX_TOKENS_PER_MINUTE = 8_500
DEFAULT_MAX_REQUESTS_PER_MINUTE = 2.5

WINDOW_SECONDS = 60.0

# Voyage counts tokens, not characters. Four characters per token is the usual
# rough figure for English prose and is deliberately conservative here: a low
# estimate would under-count and walk into the ceiling.
CHARS_PER_TOKEN = 3.5


def estimate_tokens(texts: list[str]) -> int:
    return int(sum(len(t) for t in texts) / CHARS_PER_TOKEN) + 1


class RateBudget:
    def __init__(
        self,
        max_tokens_per_minute: int = DEFAULT_MAX_TOKENS_PER_MINUTE,
        max_requests_per_minute: float = DEFAULT_MAX_REQUESTS_PER_MINUTE,
    ):
        # A budget admitting fewer than one request per window can never satisfy
        # `request_ok`, so `acquire` spins forever. Silent hangs are the worst
        # failure a pacer can have: the run looks alive and never progresses,
        # which is exactly what the 429 backoff already looked like.
        if max_requests_per_minute < 1:
            raise ValueError(
                f"max_requests_per_minute must be at least 1, got "
                f"{max_requests_per_minute}: a lower value can never admit a request"
            )
        self.max_tokens = max_tokens_per_minute
        self.max_requests = max_requests_per_minute
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > WINDOW_SECONDS:
            self._events.popleft()

    def acquire(self, tokens: int) -> float:
        """Block until this request fits both budgets. Returns seconds waited."""
        waited = 0.0
        while True:
            now = time.monotonic()
            self._prune(now)
            used_tokens = sum(t for _, t in self._events)
            used_requests = len(self._events)

            token_ok = used_tokens + tokens <= self.max_tokens
            request_ok = used_requests + 1 <= self.max_requests

            if token_ok and request_ok:
                self._events.append((now, tokens))
                return waited

            # A single request larger than the whole budget can never satisfy
            # `token_ok`, and with an empty window there is no oldest event to
            # wait on: the line below would raise IndexError. It is reachable.
            # Groq's free tier deducts the reserved max_tokens, so one `resolve`
            # call reserves almost the entire per-minute budget by itself.
            #
            # Waiting for the window to clear and then proceeding is the honest
            # behaviour: the request is as small as it can be made, and the
            # provider will accept it or refuse it on its own terms.
            if not self._events:
                if not token_ok:
                    self._events.append((now, tokens))
                    return waited
                time.sleep(0.5)
                waited += 0.5
                continue

            # Sleep until the oldest event leaves the window, which is the
            # earliest moment either budget can free up.
            oldest = self._events[0][0]
            sleep_for = max(0.5, WINDOW_SECONDS - (now - oldest) + 0.5)
            time.sleep(sleep_for)
            waited += sleep_for
