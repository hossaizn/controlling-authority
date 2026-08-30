"""Paired comparison of two triage runs over the same scenario set.

    uv run python -m eval.compare_runs <baseline.json> <arm.json>

**Why this exists rather than subtracting two macro scores.** DL-41 asks for the
deltas "as a paired comparison rather than replacing the numbers", and CLAUDE.md
already warns that precision at this scale is about one scenario. Two runs can
report an identical macro accuracy while disagreeing on a dozen scenarios, half
of them fixed and half of them broken. A delta of 0.000 then reads as "nothing
happened" when what happened is that the model became unstable in both
directions at once.

So the unit of comparison is the scenario, not the average:

- **fixed**    wrong in the baseline, right in the arm
- **broken**   right in the baseline, wrong in the arm
- **churn**    wrong in both, but wrong in a different way
- **agreed**   identical prediction, whether right or wrong

`fixed` minus `broken` is the only part that moves the score. `churn` moves
nothing and still means the run changed, which is the signal a macro delta
discards. For a temperature experiment that distinction is the whole question:
sampling variance shows up as churn and as offsetting fixed/broken pairs long
before it shows up in an average.

Runs must cover the same scenario ids and the same prompt version. Comparing
across prompt versions would attribute a prompt edit to a sampling change, so it
raises rather than reporting a number nobody can interpret.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {o["scenario_id"]: o for o in report["outcomes"]}


def compare(baseline: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Classify every scenario, then derive the aggregates from the classification.

    Derived rather than read from the two reports, so a mismatch between the
    per-scenario detail and the headline cannot survive. The reports carry their
    own `macro_accuracy`; those are echoed for reference and never used to
    compute the delta.
    """
    if baseline.get("prompt_version") != arm.get("prompt_version"):
        raise ValueError(
            "prompt versions differ "
            f"({baseline.get('prompt_version')} vs {arm.get('prompt_version')}): "
            "a prompt edit and a sampling change cannot be told apart in one delta"
        )

    left, right = _by_id(baseline), _by_id(arm)
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))[:5]
        raise ValueError(
            f"runs cover different scenarios; {len(set(left) ^ set(right))} differ, "
            f"first few: {missing}"
        )

    buckets: dict[str, list[dict[str, Any]]] = {
        "fixed": [], "broken": [], "churn": [], "agreed": []
    }
    for sid in sorted(left):
        a, b = left[sid], right[sid]
        a_ok = a["predicted"] == a["expected"]
        b_ok = b["predicted"] == b["expected"]
        row = {
            "scenario_id": sid,
            "slice_name": a["slice_name"],
            "expected": a["expected"],
            "baseline": a["predicted"],
            "arm": b["predicted"],
        }
        if a["predicted"] == b["predicted"]:
            buckets["agreed"].append(row)
        elif b_ok and not a_ok:
            buckets["fixed"].append(row)
        elif a_ok and not b_ok:
            buckets["broken"].append(row)
        else:
            buckets["churn"].append(row)

    total = len(left)
    changed = total - len(buckets["agreed"])
    return {
        "n": total,
        "baseline_macro": baseline["macro_accuracy"],
        "arm_macro": arm["macro_accuracy"],
        "macro_delta": arm["macro_accuracy"] - baseline["macro_accuracy"],
        "baseline_temperature": baseline.get("temperature"),
        "arm_temperature": arm.get("temperature"),
        "agreement_rate": len(buckets["agreed"]) / total,
        "changed": changed,
        "counts": {k: len(v) for k, v in buckets.items()},
        # The only part of `changed` that moves the score. Reported separately
        # because `net` and `changed` answer different questions and a single
        # number cannot serve both.
        "net_scenarios": len(buckets["fixed"]) - len(buckets["broken"]),
        "buckets": buckets,
    }


def format_report(result: dict[str, Any]) -> str:
    c = result["counts"]

    def temp(value: float | None) -> str:
        return "provider default" if value is None else f"temperature {value:g}"

    lines = [
        f"paired over {result['n']} scenarios",
        f"  baseline : {temp(result['baseline_temperature'])}, "
        f"macro {result['baseline_macro']:.4f}",
        f"  arm      : {temp(result['arm_temperature'])}, "
        f"macro {result['arm_macro']:.4f}",
        f"  macro delta {result['macro_delta']:+.4f}",
        "",
        f"  agreed {c['agreed']:3}   changed {result['changed']:3}   "
        f"(agreement {result['agreement_rate']:.3f})",
        f"  fixed  {c['fixed']:3}   broken  {c['broken']:3}   "
        f"churn {c['churn']:3}   net {result['net_scenarios']:+}",
    ]
    for name in ("fixed", "broken", "churn"):
        rows = result["buckets"][name]
        if not rows:
            continue
        lines += ["", f"  {name}:"]
        lines += [
            f"    {r['scenario_id']:18} {r['slice_name']:14} "
            f"want {r['expected']:9} {r['baseline']} -> {r['arm']}"
            for r in rows
        ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    baseline = json.loads(Path(sys.argv[1]).read_text())
    arm = json.loads(Path(sys.argv[2]).read_text())
    print(format_report(compare(baseline, arm)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
