"""Render a comparison across evaluation runs.

Reads the JSON reports in eval/runs/ and prints the two-by-two that settles
DL-1 (which embedding model) and DL-14 (which chunking strategy), plus the
recall gap DL-16 uses to decide whether reranking is worth building.

Reports per-slice as well as overall, because the six verified scenarios are
entirely conflict-slice: a pooled number measures difficulty as much as it
measures configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"

CONFIGS = [
    ("voyage-law", "structure"), ("voyage-law", "fixed"),
    ("voyage-general", "structure"), ("voyage-general", "fixed"),
]


def load(name: str) -> dict | None:
    path = RUNS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def overall(report: dict) -> dict[str, float]:
    scores = report["scores"]
    n = len(scores)
    r3 = sum(s["recall_at_3"] for s in scores) / n
    r10 = sum(s["recall_at_10"] for s in scores) / n
    return {
        "n": n, "recall@3": r3, "recall@10": r10,
        "mrr": sum(s["mrr"] for s in scores) / n,
        "forbidden": sum(s["forbidden_hit"] for s in scores) / n,
        "headroom": r10 - r3,
    }


def by_slice(report: dict) -> dict[str, dict[str, float]]:
    groups: dict[str, list] = {}
    for s in report["scores"]:
        groups.setdefault(s["slice"], []).append(s)
    return {
        name: {
            "n": len(g),
            "recall@10": sum(s["recall_at_10"] for s in g) / len(g),
            "recall@3": sum(s["recall_at_3"] for s in g) / len(g),
        }
        for name, g in sorted(groups.items())
    }


def main() -> None:
    reports = {(m, s): load(f"{m}_{s}") for m, s in CONFIGS}
    missing = [f"{m}/{s}" for (m, s), v in reports.items() if v is None]
    if missing:
        print(f"incomplete: still missing {missing}")
        return

    print(f"{'config':32} {'r@10':>7} {'r@3':>7} {'mrr':>7} {'forbid':>7} {'headroom':>9}")
    stats = {}
    for key in CONFIGS:
        o = overall(reports[key])
        stats[key] = o
        print(f"{key[0] + ' / ' + key[1]:32} {o['recall@10']:>7.3f} {o['recall@3']:>7.3f} "
              f"{o['mrr']:>7.3f} {o['forbidden']:>7.3f} {o['headroom']:>9.3f}")

    print("\n--- DL-1: legal vs general, chunking held constant ---")
    for strategy in ("structure", "fixed"):
        legal, general = stats[("voyage-law", strategy)], stats[("voyage-general", strategy)]
        print(f"  {strategy:10} recall@10 {100*(legal['recall@10']-general['recall@10']):+6.1f} pts"
              f" | recall@3 {100*(legal['recall@3']-general['recall@3']):+6.1f} pts")

    print("\n--- DL-14: structure vs fixed, model held constant ---")
    print("  tie-break fixed in advance: within 2 points, adopt fixed-size")
    for model in ("voyage-law", "voyage-general"):
        delta = 100 * (stats[(model, "structure")]["recall@10"]
                       - stats[(model, "fixed")]["recall@10"])
        verdict = "within tie-break -> fixed" if abs(delta) <= 2 else (
            "structure" if delta > 0 else "fixed")
        print(f"  {model:16} {delta:+6.1f} pts  -> {verdict}")

    print("\n--- DL-16: reranking headroom (recall@10 - recall@3) ---")
    print("  rule fixed in advance: build reranking only above 10 points")
    for key in CONFIGS:
        h = 100 * stats[key]["headroom"]
        verdict = "ABOVE" if h > 10 else "below"
        print(f"  {key[0]:16}/{key[1]:10} {h:>6.1f} pts  {verdict} threshold")

    best = max(stats, key=lambda k: stats[k]["recall@10"])
    print(f"\n--- by slice, best config ({best[0]} / {best[1]}) ---")
    for name, agg in by_slice(reports[best]).items():
        print(f"  {name:16} n={agg['n']:>2}  r@10={agg['recall@10']:.3f}  "
              f"r@3={agg['recall@3']:.3f}")


if __name__ == "__main__":
    main()
