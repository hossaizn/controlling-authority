"""Tests for the model caller and the two-provider adapter.

No network. The clients are built lazily and never constructed here.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.models import (
    HAIKU,
    SONNET,
    StructuredCaller,
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


# --- fake providers ---------------------------------------------------------
#
# `client_for` is replaced rather than the SDK, so no credential is read and no
# socket is opened. Each helper returns the list of kwargs the caller handed the
# provider, which is the only thing these tests assert on: what went over the
# wire, not what came back.


def models_caller():
    from agent.models import StructuredCaller as _Caller

    return _Caller()


def _install(monkeypatch, tmp_path, client) -> None:
    from agent import models

    monkeypatch.setattr(models, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(models.StructuredCaller, "client_for", lambda self, spec: client)


def _capture_anthropic(monkeypatch, tmp_path) -> list[dict]:
    sent: list[dict] = []

    class _Messages:
        def create(self, **kwargs):
            sent.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="tool_use", input={"route": "answer"})
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                stop_reason="tool_use",
            )

    _install(monkeypatch, tmp_path, SimpleNamespace(messages=_Messages()))
    return sent


def _capture_openai(monkeypatch, tmp_path) -> list[dict]:
    sent: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            sent.append(kwargs)
            call = SimpleNamespace(
                function=SimpleNamespace(arguments='{"route": "answer"}')
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[call]),
                        finish_reason="tool_calls",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    _install(monkeypatch, tmp_path, client)
    return sent


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


# --- sampling parameters in the key (DL-41) ---------------------------------


def test_an_unset_temperature_reproduces_the_pre_sampling_key() -> None:
    """**The reason 1,008 cached Haiku decisions did not have to be re-bought.**

    DL-41 assumed adding sampling to the key orphans every entry, and priced the
    fix at a full cold re-run this account cannot pay for. Encoding *unset* as
    absence rather than as a value keeps the legacy payload byte-identical.

    Pinned against an independently computed hash, not against the function, so
    a change to the payload shape fails here rather than silently invalidating a
    cache directory nobody can refill.
    """
    payload = json.dumps(
        {"model": HAIKU, "system": "sys", "user": "usr", "tool": TOOL}, sort_keys=True
    )
    expected = hashlib.sha256(payload.encode()).hexdigest()[:32]
    assert _cache_key(HAIKU, "sys", "usr", TOOL, None) == expected
    assert _cache_key(HAIKU, "sys", "usr", TOOL) == expected


def test_temperature_zero_is_a_different_key_from_unset() -> None:
    """**The falsiness trap, and it is not hypothetical.**

    `if temperature:` reads 0.0 as unset. Zero is the exact value DL-41 wants for
    `triage`, `resolve` and `verify`, so that one-character mistake would serve
    every temperature-0 call out of a default-temperature entry and report the
    experiment as a null result: no calls made, no deltas, prediction "confirmed".

    A pre-registered prediction graded against its own control arm is worse than
    no experiment, which is why this is pinned rather than left to review.
    """
    assert _cache_key(HAIKU, "s", "u", TOOL, 0.0) != _cache_key(HAIKU, "s", "u", TOOL)
    assert _cache_key(HAIKU, "s", "u", TOOL, 0) != _cache_key(HAIKU, "s", "u", TOOL)


def test_two_temperatures_never_share_a_cache_entry() -> None:
    keys = {_cache_key(HAIKU, "s", "u", TOOL, t) for t in (0.0, 0.3, 1.0)}
    assert len(keys) == 3


def test_temperature_is_omitted_from_the_request_when_unset(monkeypatch, tmp_path):
    """Sending `temperature=None` is not the same request as sending nothing.

    The cached Haiku arm was produced by a request that carried no sampling
    field, and a provider is free to treat an explicit null differently from an
    absent key. The control arm has to stay reproducible byte for byte.
    """
    sent = _capture_anthropic(monkeypatch, tmp_path)
    models_caller().call(system="s", user="u", tool=TOOL, model=HAIKU)
    assert "temperature" not in sent[0]


def test_temperature_reaches_the_anthropic_request_when_set(monkeypatch, tmp_path):
    sent = _capture_anthropic(monkeypatch, tmp_path)
    models_caller().call(system="s", user="u", tool=TOOL, model=HAIKU, temperature=0.0)
    assert sent[0]["temperature"] == 0.0


def test_temperature_reaches_the_openai_request_when_set(monkeypatch, tmp_path):
    """Both providers, because DL-41's free arms run on the OpenAI-compatible
    path and a parameter threaded to one API only would look like a null result
    on exactly the runs that are affordable."""
    sent = _capture_openai(monkeypatch, tmp_path)
    models_caller().call(
        system="s", user="u", tool=TOOL, model="openai/gpt-oss-120b", temperature=0.0
    )
    assert sent[0]["temperature"] == 0.0


def test_the_openai_request_omits_temperature_when_unset(monkeypatch, tmp_path):
    sent = _capture_openai(monkeypatch, tmp_path)
    models_caller().call(system="s", user="u", tool=TOOL, model="openai/gpt-oss-120b")
    assert "temperature" not in sent[0]


def test_the_cache_entry_records_the_temperature_it_was_made_at(monkeypatch, tmp_path):
    """So a cache directory states its own configuration. The key already
    separates the arms; this makes them auditable without rehashing 1,127 files."""
    _capture_anthropic(monkeypatch, tmp_path)
    models_caller().call(system="s", user="u", tool=TOOL, model=HAIKU, temperature=0.0)
    written = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert written["temperature"] == 0.0


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

    assert StructuredCaller().last_call is None


# --- cost and per-request accounting ----------------------------------------


def test_cost_is_computed_per_model_from_its_own_tokens() -> None:
    """Apportioning by call share reported a DIFFERENT figure for an identical
    request as soon as a second model entered the process, because the share
    moved. Cost has to come from what each model actually used."""
    usage = Usage()
    usage.add(HAIKU, 1_000_000, 0)
    assert usage.usd == pytest.approx(1.00)
    usage.add(SONNET, 1_000_000, 0)
    assert usage.usd == pytest.approx(4.00), "1.00 haiku + 3.00 sonnet"


def test_an_unpriced_model_adds_nothing_but_its_tokens_are_still_counted() -> None:
    usage = Usage()
    usage.add(HAIKU, 1_000_000, 0)
    usage.add("openai/gpt-oss-120b", 5_000_000, 1_000_000)
    assert usage.usd == pytest.approx(1.00)
    assert usage.input_tokens == 6_000_000


def test_a_haiku_only_run_is_unaffected_by_a_later_model() -> None:
    """The bug's signature: identical work, different reported cost."""
    a = Usage()
    a.add(HAIKU, 5400, 260)
    b = Usage()
    b.add(HAIKU, 5400, 260)
    b.add(SONNET, 100, 10)
    assert a.usd == pytest.approx(
        b.usd - (100 * 3.0 + 10 * 15.0) / 1_000_000
    ), "the haiku portion must not move"


def test_a_real_call_populates_the_request_scoped_usage(monkeypatch, tmp_path) -> None:
    """Driven through `call` itself, not by writing to the ContextVar by hand.

    The first version populated the variable directly, so deleting the line in
    `call` that does it left the test green: it tested the ContextVar, not the
    thing that has to use it. Eight concurrent requests each really spending
    1,000 tokens reported 1,000 through 8,000 when the API subtracted snapshots
    of shared counters.
    """
    from agent import models
    from agent.models import track_usage

    monkeypatch.setattr(models, "CACHE_DIR", tmp_path)
    caller = models.StructuredCaller(use_cache=False)
    monkeypatch.setattr(
        caller, "client_for", lambda spec: object()
    )
    monkeypatch.setattr(
        caller, "_anthropic", lambda *a, **k: ({"route": "answer"}, (120, 30))
    )
    caller.usage.add(HAIKU, 999, 999)  # pre-existing lifetime total

    with track_usage() as scoped:
        caller.call(system="s", user="u", tool=TOOL, model=HAIKU)

    assert scoped.input_tokens == 120, "the scoped total must see the call"
    assert scoped.output_tokens == 30
    assert scoped.input_tokens != caller.usage.input_tokens, "and not the lifetime one"
    assert caller.usage.input_tokens == 1119


def test_usage_spent_outside_the_block_does_not_leak_in(monkeypatch, tmp_path) -> None:
    from agent import models
    from agent.models import track_usage

    monkeypatch.setattr(models, "CACHE_DIR", tmp_path)
    caller = models.StructuredCaller(use_cache=False)
    monkeypatch.setattr(caller, "client_for", lambda spec: object())
    monkeypatch.setattr(
        caller, "_anthropic", lambda *a, **k: ({"route": "answer"}, (50, 5))
    )

    caller.call(system="s", user="u", tool=TOOL, model=HAIKU)
    with track_usage() as scoped:
        pass
    assert scoped.input_tokens == 0


def test_usage_outside_a_tracked_block_is_not_collected() -> None:
    from agent.models import _REQUEST_USAGE

    assert _REQUEST_USAGE.get() is None
