"""Pre-computed answers for the curated scenarios.

**The path most reviewers take costs nothing and returns instantly.** Six buttons
that each replay a stored run: no model call, no rate limit, no budget consumed,
no dependency on an API key being funded. That last point stopped being
hypothetical when the Anthropic account ran out of credits mid-project.

**They are recorded from real runs, not written by hand.** A hand-written "answer"
is a brochure. Replaying a captured run means the demo shows exactly what the
system produced, including a trace whose precedence rule and citations were
actually derived, and it stays honest because regenerating it requires the system
to still work.

**Staleness is the risk, so it is checked rather than hoped for.** Each record
carries the corpus snapshot and the prompt versions it was generated under. A
mismatch is reported by `stale()` rather than silently served, because a demo
showing last week's reasoning as if it were current is worse than one that admits
it needs regenerating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

STORE = Path(__file__).resolve().parent / "precomputed"

# The six the spec names, chosen so a reviewer sees something non-obvious without
# having to invent a good question. Each maps to a scenario in the golden set, so
# the demo cannot drift away from what is measured.
CURATED: dict[str, str] = {
    "straightforward": "straightforward-001",
    "ambiguous": "ambiguous-001",
    # conflict-002, not conflict-001. Both are handbook-conflict cases; this one
    # is chosen because the naive baseline gets it WRONG (it picks state; the
    # handbook's 10 paid bereavement days exceed the 5-day statutory floor, so
    # policy controls), which is what makes the baseline toggle show anything.
    # It also defeats the naive intuition in the harder direction: the system is
    # not "always prefer the law".
    #
    # conflict-001, the spec's canonical D-1 case, currently DEGRADES to a
    # referral at the verify step and is not shown. That is recorded rather than
    # hidden: see `slice_performance` in the payload, and DL-29.
    "conflict": "conflict-002",
    "superseded": "superseded-002",
    "refuse": "out-of-scope-001",
    "escalate": "out-of-scope-009",
}


@dataclass(frozen=True)
class Precomputed:
    key: str
    scenario_id: str
    question: str
    as_of: str
    employee_context: dict[str, Any]
    answer: str
    citations: list[str]
    route: str
    trace: list[dict[str, Any]]
    provenance: dict[str, str]
    # Ground truth alongside what the agent produced. A demo that shows only
    # successes is a brochure; one that shows the scored expectation next to the
    # actual result is evidence, and it stays honest without anyone policing it.
    expected: dict[str, Any] = field(default_factory=dict)

    @property
    def matched_expectation(self) -> bool:
        return self.route == self.expected.get("route")

    @property
    def slice_performance(self) -> dict[str, Any]:
        return _slice_scores().get(self.expected.get("slice"), {})

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "scenario_id": self.scenario_id,
            "question": self.question,
            "as_of": self.as_of,
            "employee_context": self.employee_context,
            "route": self.route,
            "answer": self.answer,
            "citations": self.citations,
            "trace": self.trace,
            "precomputed": True,
            "cost_usd": 0.0,
            "provenance": self.provenance,
            "expected": self.expected,
            "matched_expectation": self.matched_expectation,
            # The measured end-to-end score for this scenario's slice. A single
            # working example is not evidence that the slice works, and a demo
            # that implies otherwise is a brochure.
            "slice_performance": self.slice_performance,
        }


def _slice_scores() -> dict[str, dict[str, Any]]:
    """Per-slice end-to-end results from the last scored run.

    Read from the eval artifact rather than restated here, so the demo cannot
    quote a number the evaluation no longer produces.
    """
    run = Path(__file__).resolve().parent.parent / "eval" / "runs" / "end_to_end.json"
    if not run.exists():
        return {}
    try:
        by_slice = json.loads(run.read_text()).get("by_slice", {})
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        name: {
            "n": v["n"],
            "route_accuracy": round(v["route"], 3),
            "fully_correct": round(v["fully_correct"], 3),
        }
        for name, v in by_slice.items()
    }


def _path(key: str) -> Path:
    return STORE / f"{key}.json"


def available() -> list[str]:
    return [k for k in CURATED if _path(k).exists()]


def load(key: str) -> Precomputed | None:
    """Returns None rather than raising for an unknown or missing key.

    A missing record is a deployment gap, not a client error: the caller falls
    back to running the agent live, which is correct behaviour rather than a 500.
    """
    if key not in CURATED:
        return None
    path = _path(key)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return Precomputed(
        key=key,
        scenario_id=raw["scenario_id"],
        question=raw["question"],
        as_of=raw["as_of"],
        employee_context=raw.get("employee_context", {}),
        answer=raw["answer"],
        citations=raw.get("citations", []),
        route=raw["route"],
        trace=raw.get("trace", []),
        provenance=raw.get("provenance", {}),
        expected=raw.get("expected", {}),
    )


def save(
    key: str,
    state: dict[str, Any],
    provenance: dict[str, str],
    expected: dict[str, Any] | None = None,
) -> Path:
    """Capture a live run. Called by the generator, never by the API."""
    context = state.get("employee_context")
    record = {
        "scenario_id": CURATED[key],
        "question": state["question"],
        "as_of": str(state["as_of"]),
        "employee_context": (
            {k: v for k, v in context.model_dump().items() if v is not None}
            if context
            else {}
        ),
        "route": state.get("route"),
        "answer": state.get("answer"),
        "citations": state.get("citations", []),
        "trace": [
            {"node": e.node, "summary": e.summary, "detail": e.detail}
            for e in state.get("trace", [])
        ],
        "provenance": provenance,
        "expected": expected or {},
    }
    STORE.mkdir(parents=True, exist_ok=True)
    path = _path(key)
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def stale(current: dict[str, str]) -> list[str]:
    """Which stored records were generated under different versions.

    Checked rather than hoped for: a demo showing last week's reasoning as
    current is worse than one that says it needs regenerating.
    """
    drifted = []
    for key in available():
        record = load(key)
        if record and any(
            record.provenance.get(name) != value for name, value in current.items()
        ):
            drifted.append(key)
    return drifted


def current_provenance(corpus_snapshot: date) -> dict[str, str]:
    from agent.nodes.compose import PROMPT_VERSION as COMPOSE_V
    from agent.nodes.resolve import PROMPT_VERSION as RESOLVE_V
    from agent.nodes.triage import PROMPT_VERSION as TRIAGE_V

    return {
        "corpus_snapshot": str(corpus_snapshot),
        "triage_prompt": TRIAGE_V,
        "resolve_prompt": RESOLVE_V,
        "compose_prompt": COMPOSE_V,
    }
