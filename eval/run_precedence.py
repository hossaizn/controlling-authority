"""Precedence correctness: does the agent pick the authority that actually controls?

Scored separately from route accuracy, because the right answer from the wrong
authority is luck rather than correctness (spec, Metrics).

**The scoring rule.** A resolution is correct when every authority it treats as
defensible is in the scenario's acceptable set. Where the scenario names a single
`expected_authority`, that set has one member. Where it carries
`acceptable_authorities`, the answer is determinate while the controlling layer
is not, and any member is correct.

Subset rather than intersection, deliberately. A resolution that says "federal
and state both control" when only state does has named an authority that does not
control, and calling that correct because one of them happened to match would
reward vagueness. The converse also holds: naming one member of a genuinely
indeterminate pair is correct, because the spec says demanding a single answer
there would score a defensible system wrong.

**Run on all 57 answer scenarios regardless of how triage routed them.** This
isolates precedence. What fraction of them the agent would actually reach is a
routing question, already measured in `eval/run_routes.py`, and reported here
separately rather than folded in.

    uv run python -m eval.run_precedence
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent.build import naive_resolve
from agent.models import HAIKU, StructuredCaller, Usage
from agent.nodes.resolve import PROMPT_VERSION, make_resolve
from agent.nodes.retrieve import make_retrieve
from agent.nodes.triage import make_triage
from agent.state import Resolution, initial_state
from eval.run_retrieval import RUNS_DIR
from eval.scenarios.loader import load_all
from eval.scenarios.schema import Scenario
from retrieval.embed import get_provider
from retrieval.store import ChunkStore

ADOPTED_MODEL = "voyage-law"
ADOPTED_STRATEGY = "structure"


def acceptable_set(scenario: Scenario) -> set[str]:
    if scenario.acceptable_authorities:
        return set(scenario.acceptable_authorities)
    return {scenario.expected_authority} if scenario.expected_authority else set()


@dataclass
class PrecedenceOutcome:
    scenario_id: str
    slice_name: str
    expected: set[str]
    resolution: Resolution
    routed_to: str
    must_address: list[str] = field(default_factory=list)
    # The layer of the top-ranked passage: what a system that trusts the most
    # semantically relevant document would answer. This is the comparison the
    # whole project is built to make, so it is computed on the same run against
    # the same retrieval rather than as a separate experiment that could differ
    # for reasons nobody intended.
    naive_layer: str | None = None

    @property
    def naive_correct(self) -> bool:
        return self.naive_layer in self.expected

    @property
    def defensible(self) -> list[str]:
        return list(self.resolution.defensible)

    @property
    def correct(self) -> bool:
        got = set(self.defensible)
        return bool(got) and got <= self.expected

    @property
    def addressed_what_it_should(self) -> bool:
        named = set(self.resolution.non_controlling_to_address)
        return set(self.must_address) <= named


def run() -> dict:
    scenarios = [s for s in load_all() if s.expected_route == "answer"]
    usage = Usage()
    caller = StructuredCaller(usage=usage)

    store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)
    triage = make_triage(caller)
    retrieve = make_retrieve(store)
    resolve = make_resolve(caller)

    outcomes: list[PrecedenceOutcome] = []
    for i, s in enumerate(scenarios, 1):
        state = initial_state(s.question, s.employee_context, s.as_of_date)
        state.update(triage(state))
        state.update(retrieve(state))
        state.update(resolve(state))
        outcomes.append(
            PrecedenceOutcome(
                scenario_id=s.scenario_id,
                slice_name=s.slice,
                expected=acceptable_set(s),
                resolution=state["resolution"],
                routed_to=state["route"],
                must_address=list(s.must_address),
                # Uses the same function the demo baseline runs, rather than
                # a second copy of "take the top hit". DL-23 claimed the
                # baseline could not drift from the agent; a review pointed
                # out it could drift from itself.
                naive_layer=naive_resolve(state)["resolution"].controlling,
            )
        )
        if i % 15 == 0 or i == len(scenarios):
            print(f"  {i}/{len(scenarios)}  {usage.summary()}", flush=True)

    return report(outcomes, usage)


def report(outcomes: list[PrecedenceOutcome], usage: Usage) -> dict:
    n = len(outcomes)
    correct = sum(o.correct for o in outcomes)

    by_slice: dict[str, list[PrecedenceOutcome]] = defaultdict(list)
    for o in outcomes:
        by_slice[o.slice_name].append(o)

    naive = sum(o.naive_correct for o in outcomes)

    print()
    print("Precedence as code, against a system that trusts the top-ranked passage.")
    print()
    print(f"{'slice':16} {'n':>3} {'naive':>7} {'agent':>7} {'delta':>8}")
    for name, items in sorted(by_slice.items()):
        c = sum(o.correct for o in items)
        b = sum(o.naive_correct for o in items)
        k = len(items)
        print(f"{name:16} {k:>3} {b/k:>7.3f} {c/k:>7.3f} {100*(c-b)/k:>+7.1f}p")
    print()
    print(f"PRECEDENCE correctness   {correct}/{n} = {correct/n:.3f}")
    print(f"naive baseline           {naive}/{n} = {naive/n:.3f}")
    print(f"delta                    {100*(correct-naive)/n:+.1f} points")

    reachable = [o for o in outcomes if o.routed_to == "answer"]
    reachable_correct = sum(o.correct for o in reachable)
    print(
        f"reachable after routing  {len(reachable)}/{n}, "
        f"correct {reachable_correct}/{len(reachable)} = "
        f"{reachable_correct/max(len(reachable),1):.3f}"
    )

    with_addr = [o for o in outcomes if o.must_address]
    if with_addr:
        ok = sum(o.addressed_what_it_should for o in with_addr)
        print(f"named the beaten source  {ok}/{len(with_addr)} = {ok/len(with_addr):.3f}")

    print()
    print("rule that fired:", dict(Counter(o.resolution.rule for o in outcomes)))
    print()

    failures = [o for o in outcomes if not o.correct]
    if failures:
        print(f"{len(failures)} precedence failures:")
        for o in failures:
            got = "/".join(o.defensible) or "none"
            print(
                f"  {o.scenario_id:20} {o.slice_name:16} "
                f"want {'/'.join(sorted(o.expected)):16} got {got:16} "
                f"({o.resolution.rule})"
            )

    print()
    print(f"cost: {usage.summary()}")

    return {
        "prompt_version": PROMPT_VERSION,
        "model": HAIKU,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "precedence_accuracy": correct / n,
        "naive_baseline_accuracy": naive / n,
        "n": n,
        "by_slice": {
            name: {
                "n": len(items),
                "correct": sum(o.correct for o in items),
                "accuracy": sum(o.correct for o in items) / len(items),
                "naive_accuracy": sum(o.naive_correct for o in items) / len(items),
            }
            for name, items in sorted(by_slice.items())
        },
        "reachable_after_routing": len(reachable),
        "rules_fired": dict(Counter(o.resolution.rule for o in outcomes)),
        "failures": [
            {
                "scenario_id": o.scenario_id,
                "slice": o.slice_name,
                "expected": sorted(o.expected),
                "got": o.defensible,
                "rule": o.resolution.rule,
            }
            for o in failures
        ],
        "usage": {
            "calls": usage.calls,
            "cached": usage.cached,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "usd": round(usage.usd, 4),
        },
    }


def main() -> int:
    result = run()
    path = Path(RUNS_DIR) / f"precedence_{PROMPT_VERSION}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
