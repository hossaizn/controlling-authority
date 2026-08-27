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

from agent.citations import mentions

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
    # The supersession pair: superseded-001 and -002 are word-identical
    # questions with identical employee context, differing ONLY in as_of_date
    # (2023 against 2026). Shown side by side, they demonstrate point-in-time
    # answering better than any explanation, and they are the same pair that
    # proved in DL-21 that the date cannot be recovered from the question.
    "superseded_before": "superseded-001",
}

#: Scenarios that are the same question at two dates. The spec asks for one.
SUPERSESSION_PAIR = ("superseded_before", "superseded")


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
    # The resolution itself, not just the answer. Without it the baseline
    # comparison has nothing to compare: the delta the demo exists to show is
    # WHICH AUTHORITY each arm selects.
    controlling_authority: str | None
    defensible_authorities: list[str]
    precedence_rule: str | None
    trace: list[dict[str, Any]]
    provenance: dict[str, str]
    # Ground truth alongside what the agent produced. A demo that shows only
    # successes is a brochure; one that shows the scored expectation next to the
    # actual result is evidence, and it stays honest without anyone policing it.
    expected: dict[str, Any] = field(default_factory=dict)

    @property
    def matched_expectation(self) -> bool:
        """Route AND authority AND required citations, not route alone.

        The page labels this "matches ground truth" in green. Checking only the
        route would paint a run green that reached the right conclusion from the
        wrong authority, which the spec calls luck rather than correctness, or
        that omitted a citation the scenario requires.
        """
        if self.route != self.expected.get("route"):
            return False
        acceptable = acceptable_from(self.expected)
        if acceptable:
            defensible = set(self.defensible_authorities) or (
                {self.controlling_authority} if self.controlling_authority else set()
            )
            if not defensible or not defensible <= acceptable:
                return False
        required = self.expected.get("required_citations") or []
        answer = self.answer or ""
        return all(mentions(answer, c) for c in required)

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
            "controlling_authority": self.controlling_authority,
            "defensible_authorities": self.defensible_authorities,
            "precedence_rule": self.precedence_rule,
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


SCORES = STORE / "_slice_scores.json"


def _slice_scores() -> dict[str, dict[str, Any]]:
    """Per-slice end-to-end results, from a snapshot committed beside the records.

    **This read `eval/runs/end_to_end.json` and that directory is gitignored**,
    so every payload reported an empty `slice_performance` in any deployment and
    the honesty argument in DL-29 held only on the machine that generated it.
    Worse, it failed silently: a missing file returned `{}` and all tests passed.

    The snapshot is written by `api/generate_precomputed.py` from the eval
    artifact, in the same pass that captures the runs, so the numbers cannot
    describe a different run from the answers they sit beside. A missing
    snapshot now raises there rather than degrading here.
    """
    if not SCORES.exists():
        return {}
    try:
        return json.loads(SCORES.read_text()).get("by_slice", {})
    except (json.JSONDecodeError, OSError):
        return {}


def overall_scores() -> dict[str, Any]:
    if not SCORES.exists():
        return {}
    try:
        return json.loads(SCORES.read_text()).get("overall", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _path(key: str, arm: str = "agent") -> Path:
    """One file per (scenario, arm).

    The baseline toggle is the demo's central comparison, so it has to work with
    no funded key. Storing only the agent arm would mean the one screen the spec
    calls "the entire argument" was the one screen that needed money.
    """
    suffix = "" if arm == "agent" else f".{arm}"
    return STORE / f"{key}{suffix}.json"


def available() -> list[str]:
    return [k for k in CURATED if _path(k).exists()]


def has_baseline(key: str) -> bool:
    return _path(key, "baseline").exists()


def load(key: str, arm: str = "agent") -> Precomputed | None:
    """Returns None rather than raising for an unknown or missing key.

    A missing record is a deployment gap, not a client error: the caller falls
    back to running the agent live, which is correct behaviour rather than a 500.
    """
    if key not in CURATED:
        return None
    path = _path(key, arm)
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
        controlling_authority=raw.get("controlling_authority"),
        defensible_authorities=raw.get("defensible_authorities", []),
        precedence_rule=raw.get("precedence_rule"),
        trace=raw.get("trace", []),
        provenance=raw.get("provenance", {}),
        expected=raw.get("expected", {}),
    )


def save(
    key: str,
    state: dict[str, Any],
    provenance: dict[str, str],
    expected: dict[str, Any] | None = None,
    arm: str = "agent",
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
        "controlling_authority": (
            state["resolution"].controlling if state.get("resolution") else None
        ),
        "defensible_authorities": (
            list(state["resolution"].defensible) if state.get("resolution") else []
        ),
        "precedence_rule": (
            state["resolution"].rule if state.get("resolution") else None
        ),
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
    path = _path(key, arm)
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def acceptable_from(expected: dict[str, Any]) -> set[str]:
    """The authorities a scenario accepts.

    Mirrors `eval/run_precedence.acceptable_set`: a scenario carries either a
    single `expected_authority` or, where the answer is determinate and the
    controlling layer is not, a set of `acceptable_authorities`.
    """
    if expected.get("acceptable_authorities"):
        return set(expected["acceptable_authorities"])
    return {expected["authority"]} if expected.get("authority") else set()


def save_baseline(
    key: str,
    state: dict[str, Any],
    provenance: dict[str, str],
    expected: dict[str, Any] | None = None,
) -> Path:
    """Record what a system that trusts the top-ranked passage concludes.

    Resolution only. No composed answer, because the comparison the demo exists
    to make is which authority each arm selects, and that needs no model.
    """
    resolution = state.get("resolution")
    top = state["retrieved"][0] if state.get("retrieved") else None
    route = state.get("route")

    if resolution is None:
        # The baseline never reached a resolver: it took the same non-answering
        # route the agent did, because they share a triage. There is no delta,
        # and saying so is the honest thing. Inventing one here is how the demo
        # came to assert that a naive system would answer three questions it
        # actually refuses.
        record = {
            "scenario_id": CURATED[key],
            "arm": "baseline",
            "resolved": False,
            "route": route,
            "controlling_authority": None,
            "precedence_rule": None,
            "top_passage": {},
            "expected": expected or {},
            "correct": route == (expected or {}).get("route"),
            "no_delta_reason": (
                "The baseline shares the agent's triage, and only the answering "
                f"path reaches a resolver. Both decline here ({route}), so "
                "precedence never comes into it."
            ),
            "provenance": provenance,
        }
    else:
        record = {
            "scenario_id": CURATED[key],
            "arm": "baseline",
            "resolved": True,
            "route": route,
            "controlling_authority": resolution.controlling,
            "precedence_rule": (
                "none: the top-ranked passage was taken as authoritative"
            ),
            "top_passage": {
                "citation": top.citation if top else None,
                "authority_layer": top.authority_layer if top else None,
                "heading": top.heading if top else None,
            },
            "expected": expected or {},
            "correct": resolution.controlling in acceptable_from(expected or {}),
            "provenance": provenance,
        }
    STORE.mkdir(parents=True, exist_ok=True)
    path = _path(key, "baseline")
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def load_baseline(key: str) -> dict[str, Any] | None:
    if key not in CURATED:
        return None
    path = _path(key, "baseline")
    if not path.exists():
        return None
    return json.loads(path.read_text())


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
