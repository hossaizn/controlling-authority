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
    # **Neutralise the pacer.** `_BUDGETS` is module state shared across tests,
    # so reservations accumulate and a later test blocks on a rolling window it
    # did not fill. That turned this file from 0.5s into 61s, which is how a
    # fast suite quietly stops being run.
    for budget in models._BUDGETS.values():
        monkeypatch.setattr(budget, "acquire", lambda tokens: 0.0)


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


# --- output budget, and the two failures it produces (DL-42) -----------------


def _no_tool_call(monkeypatch, tmp_path, finish_reason):
    class _Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(tool_calls=[]),
                    finish_reason=finish_reason)],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            )

    _install(monkeypatch, tmp_path,
             SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))


def test_running_out_of_output_budget_is_not_reported_as_a_tool_choice_failure(
    monkeypatch, tmp_path
):
    """**Conflating these two sends you to the wrong fix.**

    `finish_reason=length` means the model honoured `tool_choice` and was still
    thinking when the budget ran out. The old message accused the provider of
    ignoring forced tool calls, and the response to that is to drop the
    provider. The actual response is to raise one number.

    Gemini 3.6-flash hit this on every `resolve` prompt, and the misdiagnosis
    cost a run before the finish reason was read.
    """
    _no_tool_call(monkeypatch, tmp_path, "length")
    with pytest.raises(RuntimeError, match="ran out of output budget"):
        models_caller().call(system="s", user="u", tool=TOOL, model="gemini-3.6-flash")


def test_the_budget_error_names_the_variable_that_fixes_it(monkeypatch, tmp_path):
    """An error that describes a problem without naming its lever gets read as
    fatal. This one has to say which knob to turn."""
    _no_tool_call(monkeypatch, tmp_path, "length")
    with pytest.raises(RuntimeError, match="OPEN_MODEL_REASONING_TOKENS"):
        models_caller().call(system="s", user="u", tool=TOOL, model="gemini-3.6-flash")


def test_a_genuine_tool_choice_failure_still_says_so(monkeypatch, tmp_path):
    """The invariant this project will not give up. A provider that answers in
    prose fails loudly rather than being parsed, and that message must survive
    the new branch above it."""
    _no_tool_call(monkeypatch, tmp_path, "stop")
    with pytest.raises(RuntimeError, match="cannot honour tool_choice"):
        models_caller().call(system="s", user="u", tool=TOOL, model="some-model")


def test_the_reasoning_budget_defaults_to_the_groq_value() -> None:
    """1,024 is Groq's constraint, not a property of reasoning. It stays the
    default so every cached decision and published number stands; a provider
    with room raises it by environment rather than by code."""
    from agent.models import REASONING_HEADROOM

    assert REASONING_HEADROOM == 1024


# --- DL-42: the reservation is part of the key -------------------------------


def test_an_unset_reservation_reproduces_the_pre_dl42_key() -> None:
    """The same absence encoding that made DL-41 free, reused.

    A call whose reservation matches what the old code would have computed
    hashes exactly as it did before, so every cached decision stays reachable.
    Pinned against an independently computed hash rather than against the
    function, so a change to the payload shape fails here.
    """
    payload = json.dumps(
        {"model": HAIKU, "system": "sys", "user": "usr", "tool": TOOL}, sort_keys=True
    )
    expected = hashlib.sha256(payload.encode()).hexdigest()[:32]
    assert _cache_key(HAIKU, "sys", "usr", TOOL, None, None) == expected


def test_a_changed_reservation_is_a_different_key() -> None:
    """**The defect DL-42 names.** The reservation changes what the model
    returns. Two runs at different budgets sharing one entry means a later run
    serves a decision made under a configuration it is not using."""
    base = _cache_key(HAIKU, "s", "u", TOOL)
    assert _cache_key(HAIKU, "s", "u", TOOL, None, 2048) != base
    assert _cache_key(HAIKU, "s", "u", TOOL, None, 2048) != _cache_key(
        HAIKU, "s", "u", TOOL, None, 4096
    )


def test_sampling_and_reservation_are_independent_in_the_key() -> None:
    """Four configurations, four keys. Folding them together would let a
    temperature change mask a budget change."""
    keys = {
        _cache_key(HAIKU, "s", "u", TOOL, t, r)
        for t in (None, 0.0)
        for r in (None, 2048)
    }
    assert len(keys) == 4


def test_the_default_asking_nodes_keep_their_reservation() -> None:
    """**Why DL-42's fix cost nothing.**

    `triage`, `compose` and `verify` all ask for the default 1,024, which is
    also `REASONING_HEADROOM`, so `max(max_tokens, headroom)` returns exactly
    what the old `min(headroom, available)` returned. Their cached decisions,
    including both arms of DL-41's published comparison, still hit.
    """
    from agent.models import legacy_reservation, openai_reservation

    budget, _, available = openai_reservation("s" * 300, "u" * 8000, 1024)
    assert budget == legacy_reservation(available)


def test_resolve_finally_gets_the_budget_it_has_always_asked_for() -> None:
    """`_openai` accepted `max_tokens` and discarded it, so `resolve` asked for
    2,048 and silently received 1,024. That is the half of DL-42 that was a
    plain bug rather than a cache-correctness question."""
    from agent.models import legacy_reservation, openai_reservation

    budget, _, available = openai_reservation("s" * 300, "u" * 8000, 2048)
    assert budget == 2048
    assert budget != legacy_reservation(available)


def test_the_reservation_is_capped_by_what_fits_under_the_ceiling() -> None:
    """A caller asking for more than the ceiling allows must not produce an
    illegal request. The cap is the whole reason the ceiling variable exists."""
    from agent.models import REQUEST_TOKEN_CEILING, openai_reservation

    budget, prompt_tokens, available = openai_reservation("s", "u" * 24000, 8192)
    assert budget == available
    assert prompt_tokens + budget <= REQUEST_TOKEN_CEILING


def test_the_request_reserves_exactly_what_was_hashed(monkeypatch, tmp_path) -> None:
    """**The budget hashed into the key must be the budget reserved.**

    It used to be computed inside `_openai` and nowhere else. Computing it in
    two places would let the key and the request drift apart under any future
    edit, which is a cache that lies while every test still passes.
    """
    from agent.models import openai_reservation

    sent = _capture_openai(monkeypatch, tmp_path)
    system, user = "s" * 300, "u" * 8000
    expected, _, _ = openai_reservation(system, user, 2048)

    models_caller().call(
        system=system, user=user, tool=TOOL,
        model="openai/gpt-oss-120b", max_tokens=2048,
    )
    assert sent[0]["max_tokens"] == expected


def test_a_zero_reservation_is_not_the_same_key_as_unset() -> None:
    """The DL-41 falsiness trap again, on the other field.

    `if reservation:` reads 0 as unset. A zero budget is a degenerate call, not
    an absent one, and hashing it as absent would serve it a legacy entry. The
    same one-character mistake was caught on `temperature` and would have been
    reintroduced here.
    """
    assert _cache_key(HAIKU, "s", "u", TOOL, None, 0) != _cache_key(
        HAIKU, "s", "u", TOOL, None, None
    )


def test_the_legacy_reservation_is_frozen_and_ignores_the_configured_headroom(
    monkeypatch,
) -> None:
    """**`legacy_reservation` describes history, not configuration.**

    It has to answer "what would the old code have reserved", and the old code
    hardcoded 1,024. Deriving it from `REASONING_HEADROOM` instead looks correct
    and is identical while the default is unchanged, so a test that never sets
    the variable cannot tell them apart.

    It breaks the moment a provider needs more room. Raising
    `OPEN_MODEL_REASONING_TOKENS` for Gemini would move the baseline, and calls
    that genuinely match the historical reservation would start hashing to new
    keys, orphaning the cache the encoding exists to protect.
    """
    from agent import models

    monkeypatch.setattr(models, "REASONING_HEADROOM", 16384)
    assert models.legacy_reservation(50_000) == 1024
    assert models.legacy_reservation(500) == 500


def test_a_default_asking_openai_call_still_hits_its_pre_dl42_entry(
    monkeypatch, tmp_path
) -> None:
    """**The guarantee DL-42's fix rests on, tested end to end through `call`.**

    Found by mutation R5, which replaced the `budget != legacy_reservation(...)`
    check with an unconditional `reservation = budget`. Every OpenAI call would
    then carry a reservation in its key and orphan all 208 Groq entries,
    including both arms of DL-41's published comparison, and the whole suite
    stayed green.

    The existing test compared `openai_reservation` against `legacy_reservation`
    and agreed. That checks the two functions, not the code that decides whether
    to consult them, which is where the guarantee actually lives. Same shape as
    DL-26's finding: asserting a relationship is not asserting the behaviour.

    So this writes an entry at the pre-DL-42 key and demands a cache hit.
    """
    _capture_openai(monkeypatch, tmp_path)
    system, user = "s" * 300, "u" * 8000

    legacy_key = _cache_key("openai/gpt-oss-120b", system, user, TOOL)
    (tmp_path / f"{legacy_key}.json").write_text(
        json.dumps({"result": {"route": "answer"}})
    )

    caller = models_caller()
    result = caller.call(
        system=system, user=user, tool=TOOL, model="openai/gpt-oss-120b"
    )
    assert result == {"route": "answer"}
    assert caller.usage.cached == 1
    assert caller.usage.calls == 0


def test_a_resolve_sized_openai_call_must_not_hit_the_pre_dl42_entry(
    monkeypatch, tmp_path
) -> None:
    """The other half. `resolve` genuinely changed, from a truncated 1,024 to
    the 2,048 it asked for, so serving it the old entry would be the cache lying
    about which configuration produced the answer. A fix that preserved
    everything by preserving too much would pass the test above and fail here."""
    sent = _capture_openai(monkeypatch, tmp_path)
    system, user = "s" * 300, "u" * 8000

    legacy_key = _cache_key("openai/gpt-oss-120b", system, user, TOOL)
    (tmp_path / f"{legacy_key}.json").write_text(
        json.dumps({"result": {"route": "stale"}})
    )

    caller = models_caller()
    result = caller.call(
        system=system, user=user, tool=TOOL,
        model="openai/gpt-oss-120b", max_tokens=2048,
    )
    assert result == {"route": "answer"}, "served a decision made at a different budget"
    assert caller.usage.calls == 1
    assert sent[0]["max_tokens"] == 2048
