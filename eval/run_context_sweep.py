"""DL-38: where does trimming `resolve`'s context start to cost accuracy?

Two arms, per-layer cap 3 against per-layer cap 2, on the conflict slice. The
design and its thresholds are pre-registered in `eval/decision_log.md` under
DL-38, including why the obvious capped-against-uncapped comparison was
rejected as structurally biased.

    uv run python -m eval.run_context_sweep
    uv run python -m eval.run_context_sweep --slice straightforward

**Triage is shared across arms and served from the Haiku cache.** The arms are
supposed to differ in exactly one thing. Re-running triage per arm on the open
model would spend a third of the daily budget to introduce a second difference,
and a scenario whose triage is not cached falls back to the raw question with
the state from `employee_context`, which is recorded rather than hidden.

**The budget guard exists because two runs already died mid-population.** DL-37
lost 43 of 92 scenarios to a rolling daily cap and could only report a paired
subset. Groq bills reserved `max_tokens`, not tokens produced, so the spend of a
call is knowable BEFORE making it: this stops cleanly with a stated denominator
instead of raising on scenario N.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agent.build import naive_resolve
from agent.models import HAIKU, StructuredCaller, Usage, estimate_tokens
from agent.nodes.resolve import PROMPT_VERSION, SYSTEM, _user_message, cap_per_layer, make_resolve
from agent.nodes.retrieve import make_retrieve
from agent.nodes.triage import make_triage
from agent.state import initial_state
from eval.run_precedence import (
    ADOPTED_MODEL,
    ADOPTED_STRATEGY,
    PrecedenceOutcome,
    acceptable_set,
)
from eval.run_retrieval import RUNS_DIR
from eval.scenarios.loader import load_all
from retrieval.embed import get_provider
from retrieval.store import ChunkStore

# Groq free tier, none of it in the response headers (CLAUDE.md).
DAILY_TOKEN_CAP = 200_000
RESOLVE_MAX_TOKENS = 2048
# Leave room for the retries the tool-contract failures in DL-37 will cost.
BUDGET_HEADROOM = 15_000

ARMS = (3, 2)


def open_model_id() -> str:
    import os

    return os.environ.get("OPEN_MODEL_ID", "openai/gpt-oss-120b")


def prepared(scenarios, store, cache_caller):
    """Triage + retrieve once per scenario, shared by every arm."""
    triage = make_triage(cache_caller, HAIKU)
    retrieve = make_retrieve(store)
    out = []
    for s in scenarios:
        state = initial_state(s.question, s.employee_context, s.as_of_date)
        triaged = True
        try:
            state.update(triage(state))
        except Exception:
            triaged = False
            state["jurisdiction"] = s.employee_context.state
            state["route"] = "answer"
        state.update(retrieve(state))
        out.append((s, state, triaged))
    return out


def cost_of(state, cap) -> int:
    """What this call will be billed, BEFORE making it.

    Reserved `max_tokens` is added because Groq deducts the reservation rather
    than the tokens produced, so a call's cost is knowable in advance. That is
    the whole reason a pre-flight guard is possible here at all.
    """
    trimmed = {**state, "retrieved": cap_per_layer(state["retrieved"], cap)}
    return (
        estimate_tokens([SYSTEM, _user_message(trimmed)]) + RESOLVE_MAX_TOKENS
    )


# A 429 is not a measurement. The proactive pacer prices a call from
# `estimate_tokens`, which is characters/4, and the provider counts real tokens;
# where the estimate runs low the pacer lets a call through that the bucket
# cannot take. The first run of this sweep lost 3 of its first 8 scenarios that
# way, and a scenario missing from one arm leaves the paired comparison, which
# is the only comparison DL-38 pre-registered as valid.
#
# So a rate-limited call WAITS rather than being recorded as a skip. Only a
# scenario that fails for a reason other than throttling, or that runs out of
# retries, is reported missing.
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_SLEEP_SECONDS = 65


def is_rate_limit(exc: Exception) -> bool:
    """Matched on type NAME, not by importing the provider's exception class.

    `agent/models.py` deliberately speaks to any OpenAI-compatible endpoint, so
    binding this to one SDK's class would make the retry silently stop working
    the moment the provider changes.
    """
    if type(exc).__name__ == "RateLimitError":
        return True
    return "rate limit" in str(exc).lower() or "429" in str(exc)


def resolve_with_retry(resolve, state, scenario_id: str, sleeper=time.sleep):
    last: Exception | None = None
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return resolve(state)
        except Exception as exc:
            if not is_rate_limit(exc):
                raise
            last = exc
            if attempt < RATE_LIMIT_RETRIES - 1:
                print(f"      throttled on {scenario_id}, waiting "
                      f"{RATE_LIMIT_SLEEP_SECONDS}s "
                      f"(attempt {attempt + 1}/{RATE_LIMIT_RETRIES})", flush=True)
                sleeper(RATE_LIMIT_SLEEP_SECONDS)
    raise last if last else RuntimeError("unreachable")


def affordable(spent: int, price: int, cap: int = DAILY_TOKEN_CAP,
               headroom: int = BUDGET_HEADROOM) -> bool:
    """Whether this call fits without spending into the reserve.

    Checked BEFORE the call, not after. DL-37 lost 43 of 92 scenarios because
    the cap was discovered by being hit, which raises mid-population and leaves
    a run that cannot be compared to anything.
    """
    return spent + price <= cap - headroom


def main() -> int:
    slice_name = "conflict"
    if "--slice" in sys.argv:
        slice_name = sys.argv[sys.argv.index("--slice") + 1]

    model = open_model_id()
    scenarios = [
        s for s in load_all()
        if s.expected_route == "answer" and s.slice == slice_name
    ]
    print(f"{len(scenarios)} {slice_name} scenarios, arms {ARMS}, model {model}\n")

    store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)
    cache_usage = Usage()
    work = prepared(scenarios, store, StructuredCaller(usage=cache_usage))
    untriaged = [s.scenario_id for s, _, ok in work if not ok]
    if untriaged:
        print(f"triage not cached for {len(untriaged)}: {', '.join(untriaged)}")
        print("  (fell back to the raw question; identical across arms)\n")

    for cap in ARMS:
        print(f"cap {cap}: {sum(cost_of(st, cap) for _, st, _ in work):,} tokens")
    print(f"daily cap {DAILY_TOKEN_CAP:,}, headroom {BUDGET_HEADROOM:,}\n")

    usage = Usage()
    caller = StructuredCaller(usage=usage)
    spent = 0
    results: dict[int, list[PrecedenceOutcome]] = {}
    skipped: dict[int, list[dict]] = defaultdict(list)

    for cap in ARMS:
        resolve = make_resolve(caller, model, passage_cap=cap)
        outcomes: list[PrecedenceOutcome] = []
        print(f"--- cap {cap} " + "-" * 50)
        for i, (s, base_state, _) in enumerate(work, 1):
            price = cost_of(base_state, cap)
            if not affordable(spent, price):
                skipped[cap].append({"scenario_id": s.scenario_id,
                                     "cause": "daily budget guard"})
                continue
            state = dict(base_state)
            try:
                state.update(resolve_with_retry(resolve, state, s.scenario_id))
                spent += price
            except Exception as exc:
                skipped[cap].append({"scenario_id": s.scenario_id,
                                     "cause": f"{type(exc).__name__}: {str(exc)[:120]}"})
                print(f"  {i:3}/{len(work)} SKIP {s.scenario_id:16} "
                      f"{type(exc).__name__}", flush=True)
                continue
            outcomes.append(
                PrecedenceOutcome(
                    scenario_id=s.scenario_id,
                    slice_name=s.slice,
                    expected=acceptable_set(s),
                    resolution=state["resolution"],
                    routed_to=state.get("route", "answer"),
                    must_address=list(s.must_address),
                    naive_layer=naive_resolve(state)["resolution"].controlling,
                )
            )
            mark = "ok " if outcomes[-1].correct else "MISS"
            print(f"  {i:3}/{len(work)} {mark} {s.scenario_id:16} "
                  f"spent {spent:,}", flush=True)
        results[cap] = outcomes

    print("\n" + "=" * 62)
    print(f"{'arm':>6} {'scored':>7} {'correct':>8} {'accuracy':>9} {'skipped':>8}")
    for cap in ARMS:
        o = results[cap]
        c = sum(x.correct for x in o)
        acc = f"{c/len(o):.3f}" if o else "n/a"
        print(f"cap {cap:>2} {len(o):>7} {c:>8} {acc:>9} {len(skipped[cap]):>8}")

    both = set(x.scenario_id for x in results[ARMS[0]]) & set(
        x.scenario_id for x in results[ARMS[1]]
    )
    print(f"\nscored in BOTH arms: {len(both)} of {len(work)}")
    if both:
        by_arm = {
            cap: {x.scenario_id: x for x in results[cap]} for cap in ARMS
        }
        hi, lo = ARMS
        paired_hi = sum(by_arm[hi][sid].correct for sid in both)
        paired_lo = sum(by_arm[lo][sid].correct for sid in both)
        print(f"  cap {hi}: {paired_hi}/{len(both)}   cap {lo}: {paired_lo}/{len(both)}"
              f"   delta {paired_lo - paired_hi:+d}")
        moved = [
            (sid, by_arm[hi][sid].correct, by_arm[lo][sid].correct)
            for sid in sorted(both)
            if by_arm[hi][sid].correct != by_arm[lo][sid].correct
        ]
        for sid, a, b in moved:
            print(f"    {sid:16} cap {hi} {'ok' if a else 'MISS'} "
                  f"-> cap {lo} {'ok' if b else 'MISS'}")

    print(f"\nspend: {usage.summary()} (est {spent:,} against the daily cap)")
    print(f"triage from cache: {cache_usage.summary()}")

    payload = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "slice": slice_name,
        "arms": list(ARMS),
        "estimated_tokens_spent": spent,
        "untriaged": untriaged,
        "arms_detail": {
            str(cap): {
                "scored": len(results[cap]),
                "correct": sum(x.correct for x in results[cap]),
                "skipped": skipped[cap],
                "per_scenario": [
                    {"scenario_id": x.scenario_id, "correct": x.correct,
                     "expected": sorted(x.expected), "got": x.defensible,
                     "rule": x.resolution.rule}
                    for x in results[cap]
                ],
            }
            for cap in ARMS
        },
    }
    path = Path(RUNS_DIR) / f"context_sweep_{slice_name}_{PROMPT_VERSION}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
