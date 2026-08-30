"""Run triage over the whole scenario set and score it.

All 92, not the 57 scoreable ones. Retrieval has nothing to measure on a refusal;
routing has everything to measure on it, and the routes with the fewest scenarios
are the ones macro-averaging makes expensive to get wrong.

    uv run python -m eval.run_triage

Re-runs are free: decisions are cached on disk by prompt and question, so the
second run of an unchanged prompt calls nothing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agent.models import HAIKU, StructuredCaller, Usage, temperature_slug
from agent.nodes.triage import PROMPT_VERSION, make_triage
from agent.state import initial_state
from eval.run_routes import format_report, score
from eval.scenarios.loader import load_all

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Fixed in advance, in the Phase 6 plan and agreed before any number existed:
# below this, the node moves from Haiku to Sonnet and the change is recorded.
UPGRADE_THRESHOLD = 0.80


def run(model: str = HAIKU, temperature: float | None = None) -> dict:
    """`temperature=None` sends no sampling parameter, which is how every number
    in `eval/decision_log.md` was produced. Pass a value to run DL-41's other arm."""
    scenarios = load_all()
    usage = Usage()
    triage = make_triage(
        StructuredCaller(usage=usage), model=model, temperature=temperature
    )

    predictions: dict[str, dict] = {}
    for i, s in enumerate(scenarios, 1):
        state = initial_state(s.question, s.employee_context, s.as_of_date)
        predictions[s.scenario_id] = triage(state)
        if i % 20 == 0 or i == len(scenarios):
            print(f"  {i}/{len(scenarios)}  {usage.summary()}", flush=True)

    report = score(scenarios, predictions)
    print()
    print(format_report(report))
    print()
    print(f"cost: {usage.summary()}")

    if report.failures():
        print()
        print(f"{len(report.failures())} routing failures:")
        for o in report.failures():
            print(f"  {o.scenario_id:18} {o.slice_name:14} {o.expected:9} -> {o.predicted}")

    print()
    if report.macro_accuracy < UPGRADE_THRESHOLD:
        print(
            f"macro {report.macro_accuracy:.3f} is BELOW the {UPGRADE_THRESHOLD} "
            f"threshold fixed in advance: upgrade this node and record it."
        )
    else:
        print(
            f"macro {report.macro_accuracy:.3f} is at or above the "
            f"{UPGRADE_THRESHOLD} threshold fixed in advance: {model} stays."
        )

    return {
        "model": model,
        "temperature": temperature,
        "prompt_version": PROMPT_VERSION,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "macro_accuracy": report.macro_accuracy,
        "micro_accuracy": report.micro_accuracy,
        "per_route": report.per_route(),
        "over_clarification_rate": report.over_clarification_rate,
        "under_clarification_rate": report.under_clarification_rate,
        "missing_fact_accuracy": report.missing_fact_accuracy,
        "confusion": report.confusion(),
        "usage": {
            "calls": usage.calls,
            "cached": usage.cached,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "usd": round(usage.usd, 4),
        },
        "outcomes": [asdict(o) for o in report.outcomes],
    }


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else HAIKU
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else None
    report = run(model, temperature)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Model ids contain slashes ("openai/gpt-oss-120b"), which turn a filename
    # into a nonexistent subdirectory. The run completed and printed its
    # results; only the save failed, with an exit code that read as a dead
    # experiment. run_precedence and run_end_to_end already slugify.
    slug = model.replace("/", "_")
    path = RUNS_DIR / f"triage_{slug}_{PROMPT_VERSION}{temperature_slug(temperature)}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
