"""Tests for the model caller and the two-provider adapter.

No network. The clients are built lazily and never constructed here.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from agent.models import (
    HAIKU,
    SONNET,
    Usage,
    _cache_key,
    as_openai_tool,
    parse_arguments,
    spec_for,
)

TOOL = {
    "name": "triage",
    "description": "Record a routing decision.",
    "input_schema": {
        "type": "object",
        "properties": {"route": {"type": "string", "enum": ["answer", "refuse"]}},
        "required": ["route"],
    },
}


# --- the cache must survive the adapter -------------------------------------


def test_the_haiku_cache_key_is_unchanged_by_the_adapter() -> None:
    """**The one thing this refactor could not break.**

    Every cached decision is keyed by the exact model string, and 500-odd of them
    are Haiku's arm of DL-24's comparison. If the key changed, the control arm
    would have to be re-run to be reported, and it cannot be: the account is out
    of credits. This pins the hash against a value computed independently.
    """
    payload = json.dumps(
        {"model": HAIKU, "system": "sys", "user": "usr", "tool": TOOL}, sort_keys=True
    )
    expected = hashlib.sha256(payload.encode()).hexdigest()[:32]
    assert _cache_key(HAIKU, "sys", "usr", TOOL) == expected


def test_two_models_never_share_a_cache_entry() -> None:
    """Serving one model's decision from another's cache would silently make the
    comparison compare a model against itself."""
    a = _cache_key(HAIKU, "sys", "usr", TOOL)
    b = _cache_key("llama-3.3-70b-versatile", "sys", "usr", TOOL)
    assert a != b


def test_editing_the_tool_invalidates_the_key() -> None:
    changed = {**TOOL, "input_schema": {"type": "object", "properties": {}}}
    assert _cache_key(HAIKU, "s", "u", TOOL) != _cache_key(HAIKU, "s", "u", changed)


# --- provider dispatch ------------------------------------------------------


def test_claude_models_route_to_the_anthropic_api() -> None:
    for model in (HAIKU, SONNET):
        spec = spec_for(model)
        assert spec.api == "anthropic"
        assert spec.key_env == "ANTHROPIC_API_KEY"


def test_anything_else_routes_to_an_openai_compatible_endpoint() -> None:
    """Dispatch is on the id, not a registry, so a new open model needs no code
    change: DL-24 pre-registered the comparison without naming one."""
    for model in ("llama-3.3-70b-versatile", "qwen3-32b", "gpt-oss-120b"):
        spec = spec_for(model)
        assert spec.api == "openai"
        assert spec.key_env == "OPEN_MODEL_API_KEY"
        assert spec.base_url


def test_the_anthropic_path_needs_no_open_model_credential() -> None:
    """Running the Haiku arm must not require a key for the other provider."""
    assert spec_for(HAIKU).key_env != spec_for("llama-3.3-70b-versatile").key_env


# --- tool translation -------------------------------------------------------


def test_the_json_schema_is_carried_across_unchanged() -> None:
    """Only the wrapper differs between the two APIs. Translating rather than
    keeping two copies of every tool is the whole point: two definitions of one
    contract is the drift this project keeps finding."""
    converted = as_openai_tool(TOOL)
    assert converted["type"] == "function"
    assert converted["function"]["name"] == "triage"
    assert converted["function"]["parameters"] is TOOL["input_schema"]


def test_a_tool_without_a_description_still_converts() -> None:
    converted = as_openai_tool({"name": "t", "input_schema": {"type": "object"}})
    assert converted["function"]["description"] == ""


# --- argument parsing -------------------------------------------------------


def test_plain_json_arguments_parse() -> None:
    assert parse_arguments('{"route": "answer"}', "m") == {"route": "answer"}


def test_fenced_arguments_are_recovered() -> None:
    """A fence should never appear in a function-call payload and sometimes
    does. Failing a 92-scenario run on a stray fence is the wrong trade."""
    fenced = '```json\n{"route": "refuse"}\n```'
    assert parse_arguments(fenced, "m") == {"route": "refuse"}


def test_a_bare_fence_without_a_language_is_recovered() -> None:
    assert parse_arguments('```\n{"route": "answer"}\n```', "m") == {"route": "answer"}


def test_arguments_that_are_not_an_object_raise() -> None:
    """`"123"` parses to an int and `"null"` to None. The signature promised a
    dict while the function could return anything JSON expresses, so a malformed
    response reached callers as a value they would subscript and fail on far
    from the cause."""
    for payload in ("123", "null", '"answer"', "[1, 2, 3]", "true"):
        with pytest.raises(RuntimeError, match="not an object"):
            parse_arguments(payload, "qwen3-32b")


def test_a_fenced_non_object_also_raises() -> None:
    with pytest.raises(RuntimeError, match="not an object"):
        parse_arguments('```json\n[1, 2]\n```', "qwen3-32b")


def test_unparseable_arguments_raise_with_the_model_named() -> None:
    """Loud, and naming the provider, because this is the failure that says a
    provider cannot honour the contract."""
    with pytest.raises(RuntimeError, match="qwen3-32b"):
        parse_arguments("I think the answer is refuse", "qwen3-32b")


# --- cost accounting --------------------------------------------------------


def test_a_free_model_costs_nothing_but_still_reports_tokens() -> None:
    """The dollar figure is derived from a table that can go stale; the token
    counts come from the API and are the measured quantity. So an unpriced model
    reports zero dollars and real tokens rather than being hidden."""
    usage = Usage()
    usage.add("llama-3.3-70b-versatile", 1000, 200)
    assert usage.usd == 0.0
    assert usage.input_tokens == 1000
    assert "1,000 in / 200 out" in usage.summary()


def test_a_cache_hit_is_counted_separately_from_a_call() -> None:
    usage = Usage()
    usage.add(HAIKU, 100, 10)
    usage.hit()
    assert usage.calls == 1
    assert usage.cached == 1


# --- per-call usage, and what a cache hit must NOT report --------------------


def test_a_cache_hit_clears_the_last_call_usage(monkeypatch, tmp_path) -> None:
    """A cache hit spent nothing.

    Leaving `last_call` holding the previous call's tokens makes every cached
    scenario report real cost, so an all-cache eval run bills for work that
    never happened. The tracing test covers the symptom; this pins the source.
    """
    import json as _json

    from agent import models

    monkeypatch.setattr(models, "CACHE_DIR", tmp_path)
    caller = models.StructuredCaller()
    caller.last_call = {"model": HAIKU, "input_tokens": 5400, "output_tokens": 260}

    key = _cache_key(HAIKU, "sys", "usr", TOOL)
    (tmp_path / f"{key}.json").write_text(_json.dumps({"result": {"route": "answer"}}))

    result = caller.call(system="sys", user="usr", tool=TOOL, model=HAIKU)
    assert result == {"route": "answer"}
    assert caller.last_call is None
    assert caller.usage.cached == 1
    assert caller.usage.calls == 0


def test_last_call_starts_empty() -> None:
    from agent.models import StructuredCaller

    assert StructuredCaller().last_call is None
