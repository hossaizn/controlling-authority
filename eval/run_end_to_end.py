"""End to end: all 92 scenarios through the whole graph.

Every earlier eval isolates one component by holding the others at their correct
values. This one holds nothing. A scenario mis-routed by triage never retrieves,
never resolves and never composes, and it is scored as the failure it is.

**Reported beside the isolated numbers, not instead of them.** The gap between
them is what the earlier measurements were buying with their assumptions, and it
is a finding rather than an embarrassment: a system at 0.877 on precedence given
correct routing and lower end to end tells you exactly where the next week goes.

    uv run python -m eval.run_end_to_end
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent.build import build_agent
from agent.citations import mentions
from agent.models import HAIKU, StructuredCaller, Usage
from agent.nodes.compose import PROMPT_VERSION as COMPOSE_VERSION
from agent.state import initial_state, nodes_visited
from eval.run_precedence import ADOPTED_MODEL, ADOPTED_STRATEGY, acceptable_set
from eval.run_routes import score as score_routes
from eval.scenarios.loader import load_all
from eval.scenarios.schema import Scenario
from retrieval.embed import get_provider
from retrieval.store import ChunkStore


@dataclass
class EndToEnd:
    scenario: Scenario
    final: dict
    path: list[str] = field(default_factory=list)

    @property
    def route(self) -> str:
        return self.final.get("route") or "answer"

    @property
    def route_correct(self) -> bool:
        return self.route == self.scenario.expected_route

    @property
    def answer(self) -> str:
        return self.final.get("answer") or ""

    @property
    def verified(self) -> bool:
        """Whether the answer is grounded, where grounding is a meaningful question.

        `clarify`, `refuse` and `escalate` never reach `verify`, because they
        assert no entitlement and there is nothing to ground. Scoring their
        absent verification as a failure penalised every correct refusal: it put
        `out_of_scope` at route 1.000 and fully-correct 0.000 simultaneously,
        which is the shape of a metric bug rather than a result.
        """
        v = self.final.get("verification")
        if v is None:
            return self.route in ("clarify", "refuse", "escalate")
        return v.passed

    def cites(self, citation: str) -> bool:
        """Boundary-aware, and deliberately not bracket-aware.

        A bare substring test reintroduces DL-12: `Cal. Gov. Code 12945` is a
        prefix of `Cal. Gov. Code 12945.2`, they are different statutes, and
        both are in this corpus. Requiring square brackets fixed that and
        broke the other direction, because `compose` brackets the controlling
        provision and mentions the beaten handbook in prose: the scorer then
        reported naming the beaten source as a flat zero.
        """
        return mentions(self.answer, citation)

    @property
    def required_present(self) -> bool:
        """Every citation the scenario requires appears in the answer text."""
        if not self.scenario.required_citations:
            return True
        return all(self.cites(c) for c in self.scenario.required_citations)

    @property
    def forbidden_present(self) -> bool:
        """A superseded or wrong-jurisdiction provision leaked into the answer.

        The store filters these out before retrieval, so a hit here means either
        the filter was bypassed or the model produced the citation from nowhere.
        """
        return any(self.cites(c) for c in self.scenario.forbidden_citations)

    @property
    def addressed(self) -> bool:
        return all(self.cites(c) for c in self.scenario.must_address)

    @property
    def precedence_correct(self) -> bool:
        resolution = self.final.get("resolution")
        expected = acceptable_set(self.scenario)
        if not expected:
            return True  # clarify, refuse and escalate assert no authority
        if not resolution:
            return False
        got = set(resolution.defensible)
        return bool(got) and got <= expected

    @property
    def fully_grounded(self) -> bool:
        """Verification with the judgment call counted as a failure.

        `verified` reflects the checks that BLOCK an answer. This also counts
        entailment advisories, which merely annotate it.
        """
        v = self.final.get("verification")
        if v is None:
            return self.verified
        return v.fully_grounded

    @property
    def fully_correct(self) -> bool:
        """Right route, right authority, required citations present, nothing
        forbidden, and nothing blocked it."""
        return (
            self.route_correct
            and self.precedence_correct
            and self.required_present
            and not self.forbidden_present
            and self.verified
        )

    @property
    def fully_correct_strict(self) -> bool:
        """The same, with entailment advisories counted as failures.

        **Reported as the headline.** Making entailment advisory raises
        `fully_correct` by relaxing a criterion rather than by improving the
        system, and a project that keeps catching itself at that move should not
        make it silently. The two numbers sit side by side and the stricter one
        leads.
        """
        return self.fully_correct and self.fully_grounded


def run(
    limit: int | None = None,
    model: str = HAIKU,
    verify_model: str | None = None,
) -> dict:
    scenarios = load_all()[:limit]
    usage = Usage()
    caller = StructuredCaller(usage=usage)
    store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)
    # One model across every node, which is what DL-24 compares. Mixing
    # providers per node would measure a blend rather than a model.
    agent = build_agent(
        store, caller, model=model, verify_model=verify_model or model
    )

    results: list[EndToEnd] = []
    skipped: list[tuple[str, str]] = []
    for i, s in enumerate(scenarios, 1):
        try:
            final = agent.invoke(
                initial_state(s.question, s.employee_context, s.as_of_date)
            )
        except Exception as exc:
            # **A run that dies at scenario 40 loses 40 scenarios of work.**
            # Recording the casualty and carrying on turns an outage into a
            # partial measurement with known coverage, which is usable, instead
            # of nothing. Coverage is printed so a partial run can never be
            # mistaken for a complete one.
            skipped.append((s.scenario_id, f"{type(exc).__name__}: {exc}"[:110]))
            continue
        results.append(EndToEnd(s, final, nodes_visited(final)))
        if i % 20 == 0 or i == len(scenarios):
            print(f"  {i}/{len(scenarios)}  {usage.summary()}", flush=True)

    if skipped:
        print()
        print(f"  {len(skipped)} scenario(s) could not run:")
        for sid, why in skipped[:5]:
            print(f"    {sid:20} {why}")

    return report(results, usage, model, skipped)


def _rate(items, attr: str) -> float:
    return sum(getattr(i, attr) for i in items) / len(items) if items else 0.0


def report(
    results: list[EndToEnd],
    usage: Usage,
    model: str = HAIKU,
    skipped: list[tuple[str, str]] | None = None,
) -> dict:
    skipped = skipped or []
    if skipped:
        print()
        print(
            f"  PARTIAL RUN: {len(results)}/{len(results) + len(skipped)} scenarios "
            f"scored. Rates below are over what ran."
        )
    routes = score_routes(
        [r.scenario for r in results],
        {r.scenario.scenario_id: r.final for r in results},
    )
    answers = [r for r in results if r.scenario.expected_route == "answer"]
    with_addr = [r for r in results if r.scenario.must_address]

    print()
    print("END TO END, every scenario through the whole graph")
    print()
    print(f"route accuracy, macro       {routes.macro_accuracy:.3f}")
    print(f"over-clarification          {routes.over_clarification_rate:.3f}")
    print(f"under-clarification         {routes.under_clarification_rate:.3f}")
    print()
    n_ans = len(answers)
    print(f"precedence correct          {_rate(answers, 'precedence_correct'):.3f}  (n={n_ans})")
    print(f"required citations present  {_rate(answers, 'required_present'):.3f}")
    leaked = _rate(results, "forbidden_present")
    print(f"forbidden citation leaked   {leaked:.3f}  (lower is better)")
    print(f"named the beaten source     {_rate(with_addr, 'addressed'):.3f}  (n={len(with_addr)})")
    reached = [r for r in results if r.final.get("verification") is not None]
    print(
        f"passed verification         {_rate(reached, 'verified'):.3f}  "
        f"(n={len(reached)} that reached verify)"
    )
    print()
    strict = _rate(results, "fully_correct_strict")
    relaxed = _rate(results, "fully_correct")
    print(f"FULLY CORRECT (strict)      {strict:.3f}   <- the headline")
    print(f"fully correct (blocking)    {relaxed:.3f}   entailment advisory")
    print(
        f"  difference                {relaxed - strict:+.3f}   "
        "answers shipped with a flagged claim"
    )
    print()

    by_slice: dict[str, list[EndToEnd]] = defaultdict(list)
    for r in results:
        by_slice[r.scenario.slice].append(r)
    print(f"{'slice':16} {'n':>3} {'route':>7} {'strict':>7} {'block':>7}")
    for name, items in sorted(by_slice.items()):
        print(
            f"{name:16} {len(items):>3} "
            f"{_rate(items, 'route_correct'):>7.3f} "
            f"{_rate(items, 'fully_correct_strict'):>7.3f} "
            f"{_rate(items, 'fully_correct'):>7.3f}"
        )

    leaks = [r for r in results if r.forbidden_present]
    if leaks:
        print()
        print(f"{len(leaks)} forbidden-citation leaks:")
        for r in leaks:
            print(f"  {r.scenario.scenario_id}: {r.scenario.forbidden_citations}")

    print()
    print("node paths:", dict(Counter(" > ".join(r.path) for r in results)))
    print()
    print(f"cost: {usage.summary()}")

    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "compose_prompt": COMPOSE_VERSION,
        "n": len(results),
        "attempted": len(results) + len(skipped),
        "skipped": [{"scenario_id": sid, "error": why} for sid, why in skipped],
        "route_accuracy_macro": routes.macro_accuracy,
        "over_clarification_rate": routes.over_clarification_rate,
        "under_clarification_rate": routes.under_clarification_rate,
        "precedence_correct": _rate(answers, "precedence_correct"),
        "required_citations_present": _rate(answers, "required_present"),
        "forbidden_citation_rate": _rate(results, "forbidden_present"),
        "addressed_beaten_source": _rate(with_addr, "addressed"),
        # Denominator is the scenarios that reached verify. Counting the 34
        # that never run it as passes reported 0.717 for a real rate of
        # 0.552: the same "did not run is not a pass" conflation DL-25 fixed
        # for fully_correct and left standing one line away.
        "verification_pass_rate": _rate(reached, "verified"),
        "verification_n": len(reached),
        "fully_correct": _rate(results, "fully_correct"),
        "fully_correct_strict": _rate(results, "fully_correct_strict"),
        "fully_grounded": _rate(results, "fully_grounded"),
        "by_slice": {
            name: {
                "n": len(items),
                "route": _rate(items, "route_correct"),
                "fully_correct": _rate(items, "fully_correct"),
                "fully_correct_strict": _rate(items, "fully_correct_strict"),
            }
            for name, items in sorted(by_slice.items())
        },
        "failures": [
            {
                "scenario_id": r.scenario.scenario_id,
                "slice": r.scenario.slice,
                "expected_route": r.scenario.expected_route,
                "got_route": r.route,
                "precedence_correct": r.precedence_correct,
                "required_present": r.required_present,
                "verified": r.verified,
                "path": r.path,
            }
            for r in results
            if not r.fully_correct
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
    """`run_end_to_end [model] [verify_model]`.

    A separate verify model is the architecture the spec asked for in DL-15: a
    model checking its own output shares its blind spots. Credit exhaustion made
    it the only runnable configuration, which is a convenient accident.
    """
    model = sys.argv[1] if len(sys.argv) > 1 else HAIKU
    verify_model = sys.argv[2] if len(sys.argv) > 2 else None
    result = run(model=model, verify_model=verify_model)
    slug = model.replace('/', '_')
    path = Path(__file__).resolve().parent / "runs" / f"end_to_end_{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
