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

from ingest.settings import optional, require
from retrieval.ratelimit import RateBudget, estimate_tokens

CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "model"

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-5"


def open_model() -> str:
    """The open-weights model id, chosen by environment rather than by code.

    DL-24 pre-registered the comparison without naming a model, because free-tier
    lineups change faster than this repo does. Setting `OPEN_MODEL_ID` and
    `OPEN_MODEL_BASE_URL` is enough to point at Groq, Cerebras, OpenRouter,
    Together or anything else speaking the OpenAI API, with no code change.
    """
    return optional("OPEN_MODEL_ID", "llama-3.3-70b-versatile")


@dataclass(frozen=True)
class ModelSpec:
    api: str  # "anthropic" | "openai"
    key_env: str
    base_url: str | None = None


def spec_for(model: str) -> ModelSpec:
    """Which API a model id belongs to.

    Dispatched on the id rather than on a registry key **so the cache survives**.
    Every cached decision is keyed by the exact model string, and 500-odd of them
    are Haiku's arm of the DL-24 comparison. Renaming the identifier would
    invalidate all of them and make the control arm cost money to reproduce,
    which is the one thing a comparison cannot afford.
    """
    if model.startswith("claude-"):
        return ModelSpec(api="anthropic", key_env="ANTHROPIC_API_KEY")
    return ModelSpec(
        api="openai",
        key_env="OPEN_MODEL_API_KEY",
        base_url=optional("OPEN_MODEL_BASE_URL", "https://api.groq.com/openai/v1"),
    )


# USD per million tokens. Written down rather than inferred so a wrong number is
# visible and correctable in one place; verify against the provider's pricing
# page before quoting a cost anywhere it matters. Token counts below are reported
# by the API and are the measured quantity; the dollar figure is derived.
#
# An unlisted model prices at zero, which is right for a free tier and is why the
# figure is always reported beside the raw token counts rather than alone.
PRICE_PER_MTOK = {
    HAIKU: {"input": 1.00, "output": 5.00},
    SONNET: {"input": 3.00, "output": 15.00},
}

# Per API, because the constraints differ in kind. A paid Anthropic account is
# nowhere near its ceiling and is paced only to avoid surprises; a free tier is
# genuinely rate-limited and pacing it is what keeps a 92-scenario run alive.
# Reactive backoff was tried on Voyage and died 264 chunks into 300.
_BUDGETS = {
    "anthropic": RateBudget(max_tokens_per_minute=100_000, max_requests_per_minute=40),
    "openai": RateBudget(max_tokens_per_minute=25_000, max_requests_per_minute=25),
}


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


def as_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic's tool shape into OpenAI's.

    The JSON Schema in `input_schema` is the same object in both; only the
    wrapper differs. Written as a translation rather than by keeping two copies
    of every tool definition, because two definitions of one contract is the
    drift this project keeps finding.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


def parse_arguments(raw: str, model: str) -> dict[str, Any]:
    """Tool arguments should be raw JSON. Some providers still fence them.

    Anthropic returns a parsed object, so this path only exists for the
    OpenAI-compatible side, where `arguments` is a string. A fence should never
    appear in a function-call payload and sometimes does, so it is stripped
    rather than allowed to fail a whole run.
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{model} returned tool arguments that are not JSON: {raw[:200]!r}"
        ) from exc


class StructuredCaller:
    """Calls a model and returns the arguments of a forced tool call.

    Two APIs behind one method. Anthropic and any OpenAI-compatible endpoint,
    which covers every free open-weights host worth using, so DL-24's comparison
    is a change of environment variable rather than a second implementation.

    **The forced tool call is the invariant.** It is what keeps structured output
    from degenerating into parsing prose, and a provider that cannot honour
    `tool_choice` is not usable here regardless of how it scores.
    """

    def __init__(self, usage: Usage | None = None, use_cache: bool = True):
        self._clients: dict[str, Any] = {}
        self.usage = usage or Usage()
        self.use_cache = use_cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def client_for(self, spec: ModelSpec):
        # Built on first use, not on construction, so importing this module in a
        # test that never calls a model does not require a credential.
        if spec.api not in self._clients:
            if spec.api == "anthropic":
                self._clients[spec.api] = Anthropic(api_key=require(spec.key_env))
            else:
                from openai import OpenAI

                self._clients[spec.api] = OpenAI(
                    api_key=require(spec.key_env), base_url=spec.base_url
                )
        return self._clients[spec.api]

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

        spec = spec_for(model)
        _BUDGETS[spec.api].acquire(estimate_tokens([system, user]))
        client = self.client_for(spec)

        if spec.api == "anthropic":
            result, tokens = self._anthropic(client, system, user, tool, model, max_tokens)
        else:
            result, tokens = self._openai(client, system, user, tool, model, max_tokens)

        self.usage.add(model, *tokens)

        if self.use_cache:
            path.write_text(
                json.dumps({"model": model, "key": key, "result": result}, indent=2)
            )
        return result

    def _anthropic(self, client, system, user, tool, model, max_tokens):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            # Forced. Without this the model may answer in prose and the caller
            # has to handle two shapes, which is how a parser starts guessing.
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        blocks = [b for b in response.content if b.type == "tool_use"]
        if not blocks:
            raise RuntimeError(
                f"{model} returned no tool call despite tool_choice being forced; "
                f"stop_reason={response.stop_reason}"
            )
        return blocks[0].input, (
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def _openai(self, client, system, user, tool, model, max_tokens):
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[as_openai_tool(tool)],
            # Same invariant as the Anthropic path. A provider that ignores this
            # and answers in prose fails loudly below rather than being parsed.
            tool_choice={"type": "function", "function": {"name": tool["name"]}},
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            raise RuntimeError(
                f"{model} returned no tool call despite tool_choice being forced; "
                f"finish_reason={response.choices[0].finish_reason}. "
                "A provider that cannot honour tool_choice is not usable here."
            )
        result = parse_arguments(calls[0].function.arguments, model)
        usage = response.usage
        return result, (
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )
