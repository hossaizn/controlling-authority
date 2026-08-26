"""Route accuracy, macro-averaged, plus the two failure directions around clarify.

**Macro-averaged, never micro.** The set is 57 answer, 15 clarify, 14 refuse and
6 escalate, so a system that never clarifies at all still scores 83% micro while
being useless at the hardest thing it does (DL-7). Each route contributes
equally here regardless of how many scenarios it has.

**Both clarify directions are reported.** Over-clarification is the failure users
actually experience, and an agent that always asks is trivially safe and
unusable. Under-clarification is the one a majority-class metric hides. Neither
is derivable from the macro figure alone, so both are printed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from domain import Route
from eval.scenarios.schema import Scenario

ROUTES: tuple[Route, ...] = ("answer", "clarify", "refuse", "escalate")


@dataclass
class RouteOutcome:
    scenario_id: str
    slice_name: str
    expected: Route
    predicted: Route
    expected_fact: str | None
    predicted_fact: str | None

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted

    @property
    def fact_correct(self) -> bool:
        """Only meaningful on a correctly-routed clarify. Asking the wrong
        question is not the same failure as not asking one."""
        return self.expected_fact == self.predicted_fact


@dataclass
class RouteReport:
    outcomes: list[RouteOutcome] = field(default_factory=list)

    def per_route(self) -> dict[str, dict]:
        by_route: dict[str, list[RouteOutcome]] = defaultdict(list)
        for o in self.outcomes:
            by_route[o.expected].append(o)
        return {
            route: {
                "n": len(items),
                "correct": sum(o.correct for o in items),
                "accuracy": sum(o.correct for o in items) / len(items),
            }
            for route, items in sorted(by_route.items())
            if items
        }

    @property
    def macro_accuracy(self) -> float:
        stats = self.per_route()
        return sum(s["accuracy"] for s in stats.values()) / len(stats)

    @property
    def micro_accuracy(self) -> float:
        """Reported only so the gap to macro is visible. Never used as the
        headline; see the module docstring."""
        return sum(o.correct for o in self.outcomes) / len(self.outcomes)

    @property
    def over_clarification_rate(self) -> float:
        """Asked when it should not have. The metric that shows judgment."""
        eligible = [o for o in self.outcomes if o.expected != "clarify"]
        if not eligible:
            return 0.0
        return sum(o.predicted == "clarify" for o in eligible) / len(eligible)

    @property
    def under_clarification_rate(self) -> float:
        eligible = [o for o in self.outcomes if o.expected == "clarify"]
        if not eligible:
            return 0.0
        return sum(o.predicted != "clarify" for o in eligible) / len(eligible)

    @property
    def missing_fact_accuracy(self) -> float:
        """Among clarifies that correctly fired, how often the right fact was
        named. A clarify that asks for the wrong thing wastes the turn."""
        hits = [o for o in self.outcomes if o.expected == "clarify" and o.correct]
        if not hits:
            return 0.0
        return sum(o.fact_correct for o in hits) / len(hits)

    def confusion(self) -> dict[str, dict[str, int]]:
        matrix = {e: dict.fromkeys(ROUTES, 0) for e in ROUTES}
        for o in self.outcomes:
            matrix[o.expected][o.predicted] = matrix[o.expected].get(o.predicted, 0) + 1
        return matrix

    def failures(self) -> list[RouteOutcome]:
        return [o for o in self.outcomes if not o.correct]


def score(scenarios: list[Scenario], predictions: dict[str, dict]) -> RouteReport:
    """`predictions` maps scenario_id to the triage node's returned state."""
    report = RouteReport()
    for s in scenarios:
        pred = predictions[s.scenario_id]
        report.outcomes.append(
            RouteOutcome(
                scenario_id=s.scenario_id,
                slice_name=s.slice,
                expected=s.expected_route,
                predicted=pred["route"],
                expected_fact=s.missing_fact,
                predicted_fact=pred.get("missing_fact"),
            )
        )
    return report


def format_report(report: RouteReport) -> str:
    lines = ["route accuracy, per route", ""]
    lines.append(f"{'route':10} {'n':>3} {'correct':>8} {'accuracy':>9}")
    for route, s in report.per_route().items():
        lines.append(f"{route:10} {s['n']:>3} {s['correct']:>8} {s['accuracy']:>9.3f}")
    lines.append("")
    lines.append(f"MACRO accuracy        {report.macro_accuracy:.3f}   <- the headline")
    lines.append(f"micro accuracy        {report.micro_accuracy:.3f}   (majority-weighted)")
    lines.append("")
    lines.append(f"over-clarification    {report.over_clarification_rate:.3f}")
    lines.append(f"under-clarification   {report.under_clarification_rate:.3f}")
    lines.append(f"missing-fact accuracy {report.missing_fact_accuracy:.3f}")
    lines.append("")
    lines.append("confusion (rows expected, columns predicted)")
    lines.append(f"{'':10}" + "".join(f"{r:>10}" for r in ROUTES))
    for expected, row in report.confusion().items():
        lines.append(f"{expected:10}" + "".join(f"{row[r]:>10}" for r in ROUTES))
    return "\n".join(lines)
