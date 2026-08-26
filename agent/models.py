"""Model calls: cached, structured, and accounted for.

Every node in the graph calls a model through here rather than holding its own
client, for three reasons that each cost something to learn elsewhere.

**Structured output comes from tool use, not from parsing prose.** Asking for
JSON in a prompt returns JSON wrapped in a markdown fence often enough that the
fence-stripping becomes load-bearing, and then one day the model emits prose
before the fence and the strip silently returns something unparseable. A forced
tool call is validated by the API against a schema before it reaches this code.

**Responses are cached on disk by content hash.** A model call is not a pure
function, but for evaluation it has to behave like one: re-running the scenario
set to check a change downstream must not re-pay for, or re-roll, 92 upstream
decisions. Same reasoning as the embedding cache, which turned a 66-minute run
into 5 minutes. The prompt version is part of the key, so editing a prompt
invalidates exactly the entries it should.

**Token counts are recorded, not estimated.** The usage figures come from the
API. The dollar figure is derived from a rate written down below, which is the
only part of this that can go stale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from ingest.settings import require
from retrieval.ratelimit import RateBudget, estimate_tokens

CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "model"

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-5"

# USD per million tokens. Written down rather than inferred so a wrong number is
# visible and correctable in one place; verify against Anthropic's pricing page
# before quoting a cost anywhere it matters. Token counts below are reported by
# the API and are the measured quantity; the dollar figure is derived from these.
PRICE_PER_MTOK = {
    HAIKU: {"input": 1.00, "output": 5.00},
    SONNET: {"input": 3.00, "output": 15.00},
}

# Well inside any paid tier. The binding constraint on this project is not the
# rate limit, it is not wanting a surprise, so calls are paced anyway.
_BUDGET = RateBudget(max_tokens_per_minute=100_000, max_requests_per_minute=40)


@dataclass
class Usage:
    """Accumulated across a run so a phase can report what it cost."""

    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def hit(self) -> None:
        self.cached += 1

    @property
    def usd(self) -> float:
        total = 0.0
        for model, count in self.by_model.items():
            if count and model in PRICE_PER_MTOK:
                rate = PRICE_PER_MTOK[model]
                share = count / max(sum(self.by_model.values()), 1)
                total += (
                    self.input_tokens * share * rate["input"]
                    + self.output_tokens * share * rate["output"]
                ) / 1_000_000
        return total

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.cached} served from cache), "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens, "
            f"≈${self.usd:.4f}"
        )


def _cache_key(model: str, system: str, user: str, tool: dict) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "user": user, "tool": tool},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class StructuredCaller:
    """Calls a model and returns the arguments of a forced tool call."""

    def __init__(self, usage: Usage | None = None, use_cache: bool = True):
        self._client: Anthropic | None = None
        self.usage = usage or Usage()
        self.use_cache = use_cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> Anthropic:
        # Built on first use, not on construction, so importing this module in a
        # test that never calls a model does not require a credential.
        if self._client is None:
            self._client = Anthropic(api_key=require("ANTHROPIC_API_KEY"))
        return self._client

    def call(
        self,
        system: str,
        user: str,
        tool: dict[str, Any],
        model: str = HAIKU,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        key = _cache_key(model, system, user, tool)
        path = CACHE_DIR / f"{key}.json"

        if self.use_cache and path.exists():
            self.usage.hit()
            return json.loads(path.read_text())["result"]

        _BUDGET.acquire(estimate_tokens([system, user]))

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            # Forced. Without this the model may answer in prose and the caller
            # has to handle two shapes, which is how a parser starts guessing.
            tool_choice={"type": "tool", "name": tool["name"]},
        )

        self.usage.add(
            model, response.usage.input_tokens, response.usage.output_tokens
        )

        blocks = [b for b in response.content if b.type == "tool_use"]
        if not blocks:
            raise RuntimeError(
                f"{model} returned no tool call despite tool_choice being forced; "
                f"stop_reason={response.stop_reason}"
            )
        result = blocks[0].input

        if self.use_cache:
            path.write_text(
                json.dumps({"model": model, "key": key, "result": result}, indent=2)
            )
        return result
