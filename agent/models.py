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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from ingest.settings import optional, require
from retrieval.ratelimit import RateBudget, estimate_tokens

CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "model"

# Per-request usage, so concurrent callers cannot read each other's totals.
#
# The API originally snapshotted the shared caller's counters before and after a
# request and subtracted. Under concurrency that is simply wrong: a review ran
# eight simultaneous requests each spending 1,000 tokens and they reported
# 1,000 through 8,000. A ContextVar is scoped to the request in both threadpool
# and async execution, which a shared attribute is not.
_REQUEST_USAGE: ContextVar[Usage | None] = ContextVar("request_usage", default=None)


@contextmanager
def track_usage():
    """Collect only what is spent inside this block."""
    usage = Usage()
    token = _REQUEST_USAGE.set(usage)
    try:
        yield usage
    finally:
        _REQUEST_USAGE.reset(token)

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
    # Groq's free tier: an 8,000-token bucket refilling at 8,000/minute, with the
    # reserved max_tokens deducted rather than the tokens produced. Paced at 80%
    # of the bucket: measured, pacing at 100% produced six stalls over 90s in 74
    # minutes because the SDK absorbs a 429 with a retry-after sleep.
    "openai": RateBudget(
        max_tokens_per_minute=int(optional("OPEN_MODEL_TPM", "8000")) * 80 // 100,
        max_requests_per_minute=30,
    ),
}


# Output budget for OpenAI-compatible models, and the ceiling it has to fit under.
#
# Two constraints pull against each other, and both were found by running it:
#
# 1. **Reasoning tokens come out of `max_tokens` and are emitted before the tool
#    call.** A budget sized for the answer alone truncates the call, and the
#    provider rejects it server-side with `tool_use_failed`.
# 2. **Free tiers count reserved `max_tokens` toward the per-request ceiling.**
#    Groq's free tier allows 8,000 tokens per request for gpt-oss-120b, so asking
#    for 8,192 output made every request illegal on its own, whatever the prompt.
#
# So the budget is computed per call: as much headroom as the ceiling allows once
# the prompt is paid for. `OPEN_MODEL_TPM` makes the ceiling configurable because
# it differs per provider, per model and per tier.
# Deliberately modest. Groq charges the RESERVED max_tokens against the rate
# bucket, not the tokens actually produced, so an oversized budget throttles
# throughput without buying anything: observed triage outputs were 294 and 814
# tokens including reasoning. 1,024 leaves room to think and still cuts the
# per-request reservation substantially.
REASONING_HEADROOM = 1024
# Just inside the bucket, so our token estimate disagreeing with theirs by a
# few percent cannot turn a legal request into a 413.
# **Two different limits, two variables.** These were both derived from
# OPEN_MODEL_TPM, which is wrong: the bucket is per MINUTE and the ceiling is
# per REQUEST, and they only coincide on Groq's free tier because it happens
# to set both to 8,000. Point OPEN_MODEL_TPM at a real tier (30,000) and every
# request would have been rejected as too large.
REQUEST_TOKEN_CEILING = int(
    optional("OPEN_MODEL_MAX_REQUEST_TOKENS", optional("OPEN_MODEL_TPM", "8000"))
) * 95 // 100
# Enough for a tool call with no thinking at all. Below this, the model has no
# room to answer and the run should fail loudly rather than emit truncated JSON.
MIN_OUTPUT_TOKENS = 700


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
        tin, tout = self.tokens_by_model.get(model, (0, 0))
        self.tokens_by_model[model] = (tin + input_tokens, tout + output_tokens)

    def hit(self) -> None:
        self.cached += 1

    #: Tokens per model, so cost is computed from what each model actually used
    #: rather than apportioned by call count. Apportioning reported a different
    #: figure for an identical request as soon as a second model entered the
    #: process, because the share moved.
    tokens_by_model: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def usd(self) -> float:
        total = 0.0
        for model, (tin, tout) in self.tokens_by_model.items():
            rate = PRICE_PER_MTOK.get(model)
            if rate:
                total += (tin * rate["input"] + tout * rate["output"]) / 1_000_000
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
        return _require_object(json.loads(text), raw, model)
    except json.JSONDecodeError:
        pass
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return _require_object(json.loads(text.strip()), raw, model)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{model} returned tool arguments that are not JSON: {raw[:200]!r}"
        ) from exc


def _require_object(value: Any, raw: str, model: str) -> dict[str, Any]:
    """Tool arguments are an object. `"123"` parses to an int, `"null"` to None.

    The signature promised a dict while the function could return anything JSON
    can express, so a malformed response reached callers as a value they would
    subscript and fail on far from the cause.
    """
    if not isinstance(value, dict):
        raise RuntimeError(
            f"{model} returned tool arguments that are not an object "
            f"({type(value).__name__}): {raw[:200]!r}"
        )
    return value


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
        # The most recent call's own tokens, so a node can record what IT
        # cost rather than the run total. The plan asks for per-stage cost;
        # a single aggregate on the root cannot answer which node is
        # expensive, which is the only question the number is useful for.
        self.last_call: dict[str, Any] | None = None
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
            scoped = _REQUEST_USAGE.get()
            if scoped is not None:
                scoped.hit()
            # A cache hit spent nothing. Reporting the previous call's
            # tokens here would attribute real cost to work that never
            # happened, which is how an all-cache eval run reports a bill.
            self.last_call = None
            return json.loads(path.read_text())["result"]

        spec = spec_for(model)
        client = self.client_for(spec)

        if spec.api == "anthropic":
            _BUDGETS["anthropic"].acquire(estimate_tokens([system, user]))
            result, tokens = self._anthropic(client, system, user, tool, model, max_tokens)
        else:
            result, tokens = self._openai(client, system, user, tool, model, max_tokens)

        self.usage.add(model, *tokens)
        scoped = _REQUEST_USAGE.get()
        if scoped is not None:
            scoped.add(model, *tokens)
        self.last_call = {
            "model": model,
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
        }

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
        # **Reasoning tokens are billed against `max_tokens` and are emitted
        # before the tool call.** `gpt-oss-120b` spent most of a 1024 budget
        # thinking, then produced a tool call one closing brace short, and the
        # provider rejected it server-side with `tool_use_failed` before this
        # code ever saw a payload to parse.
        #
        # The budget is raised rather than the reasoning turned down. Reducing
        # reasoning effort would make the open model cheaper and faster at
        # exactly the judgment DL-24 exists to measure, which would bias the
        # comparison in favour of the conclusion that it cannot do it.
        prompt_tokens = estimate_tokens([system, user])
        available = REQUEST_TOKEN_CEILING - prompt_tokens
        # Against the real ceiling, not the discounted one. The 5% margin
        # exists to absorb tokeniser disagreement on the RESERVATION, and
        # applying it to the floor check as well rejected prompts of
        # 6,901 to 7,300 tokens that the provider accepts.
        budget = min(REASONING_HEADROOM, available)
        headroom_at_full = (
            int(optional("OPEN_MODEL_MAX_REQUEST_TOKENS", optional("OPEN_MODEL_TPM", "8000")))
            - prompt_tokens
        )
        if headroom_at_full < MIN_OUTPUT_TOKENS:
            raise RuntimeError(
                f"{model}: a {prompt_tokens}-token prompt leaves {available} tokens "
                f"under the {REQUEST_TOKEN_CEILING} per-request ceiling, below the "
                f"{MIN_OUTPUT_TOKENS} a tool call needs. Raise "
                "OPEN_MODEL_MAX_REQUEST_TOKENS if the tier allows a larger single "
                "request, or the prompt has to shrink."
            )

        # Paced on prompt PLUS reservation, because that is what the provider
        # deducts. Pacing on the prompt alone under-counts by the whole output
        # budget and walks straight into a 429.
        _BUDGETS["openai"].acquire(prompt_tokens + budget)

        response = client.chat.completions.create(
            model=model,
            max_tokens=budget,
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
