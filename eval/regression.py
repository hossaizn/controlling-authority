"""The regression gate for Phase 6.

Written before any agent code exists, deliberately. A guard authored after
results are visible gets fitted to them.

**What it actually protects against.** Retrieval currently receives the raw
question. `triage` will replace that with a rewritten one, and if rewriting
produces worse queries, retrieval degrades. An end-to-end score can hide this
completely: better routing offsets worse retrieval and the total looks fine while
the system got worse at the thing it was already good at.

**Per slice, never overall.** An overall figure stays flat while conflict gains
fifteen points and straightforward loses fifteen. The slices are the entire
reason the scenario set is structured the way it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent / "baseline_retrieval.json"

# Tolerances fixed 2026-08-26, before the agent existed.
#
# Slices already at 1.000 cannot improve, so any movement is a regression and
# they get no allowance at all. Slices below it get one scenario's worth of
# noise, computed from their own n rather than a flat percentage, because one
# scenario is 5.9 points at n=17 and 10 points at n=10.
PERFECT_SLICE_TOLERANCE = 0.0
SCENARIOS_OF_NOISE_ALLOWED = 1

# The conflict slice is the reason Phase 6 exists. It is held to a different
# standard: not "do not regress" but "improve". Leaving it at its baseline means
# the phase did not do the one thing it was for.
MUST_IMPROVE = "conflict"

# "Improve" means at least one more scenario answered correctly, not any positive
# epsilon.
#
# Found by running the gate against real data rather than synthetic values: the
# baseline is stored rounded to four decimals, so an unchanged full-precision
# 0.72222 tests as greater than a stored 0.7222 and reported "improved by 0.0
# points, pass". A rounding artifact would have let Phase 6 clear the one bar it
# exists to clear.
MUST_IMPROVE_BY_SCENARIOS = 1


@dataclass(frozen=True)
class SliceVerdict:
    name: str
    n: int
    baseline: float
    observed: float
    tolerance: float
    passed: bool
    reason: str
    verified_n: int = 0

    @property
    def delta_points(self) -> float:
        return 100 * (self.observed - self.baseline)

    @property
    def evidence(self) -> str:
        """How much of this slice rests on ground truth that was checked against
        ingested text, rather than drafted from recall (DL-3).

        Printed rather than enforced. The gate compares a slice against itself,
        so a ground-truth error sits on both sides and cancels; what it cannot do
        is tell you that a one-scenario improvement on `conflict` may have come
        entirely from the eleven scenarios nobody has verified. That is a real
        limit on what a pass means and it belongs on the report, not in a
        footnote someone has to go looking for.
        """
        return f"{self.verified_n}/{self.n} verified"


def load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def tolerance_for(name: str, n: int, baseline: float) -> float:
    """One scenario of slack, unless the slice is already perfect."""
    if baseline >= 1.0:
        return PERFECT_SLICE_TOLERANCE
    return SCENARIOS_OF_NOISE_ALLOWED / n


def check(observed_by_slice: dict[str, float]) -> list[SliceVerdict]:
    """Compare observed per-slice recall@10 against the frozen baseline."""
    baseline = load_baseline()
    verdicts: list[SliceVerdict] = []

    for name, base in sorted(baseline["by_slice"].items()):
        n = base["n"]
        want = base["recall@10"]
        verified_n = base.get("composition", {}).get("verified", 0)
        got = observed_by_slice.get(name)

        if got is None:
            verdicts.append(SliceVerdict(
                name, n, want, 0.0, 0.0, False,
                "slice missing from results; a slice that vanishes is not a pass",
                verified_n,
            ))
            continue

        tol = tolerance_for(name, n, want)

        if name == MUST_IMPROVE:
            required = MUST_IMPROVE_BY_SCENARIOS / n
            passed = got >= want + required
            if passed:
                reason = (
                    f"improved by {100*(got-want):.1f} points, "
                    f"at least {MUST_IMPROVE_BY_SCENARIOS} scenario"
                )
            else:
                reason = (
                    f"improved by only {100*(got-want):+.1f} points; needs at least "
                    f"{100*required:.1f} (one scenario). Phase 6 exists to close this "
                    "slice, so holding is a failure rather than a hold"
                )
        else:
            passed = got >= want - tol
            if passed and got < want:
                reason = f"within tolerance ({100*(got-want):+.1f} pts, allowed {100*tol:.1f})"
            elif passed:
                reason = f"held or improved ({100*(got-want):+.1f} pts)"
            else:
                reason = (
                    f"REGRESSED {100*(want-got):.1f} points, tolerance {100*tol:.1f}"
                )

        verdicts.append(SliceVerdict(name, n, want, got, tol, passed, reason, verified_n))

    return verdicts


def format_report(verdicts: list[SliceVerdict]) -> str:
    lines = [
        f"{'slice':16} {'n':>3} {'base':>7} {'now':>7} {'delta':>8} {'evidence':>14}  verdict"
    ]
    for v in verdicts:
        mark = "pass" if v.passed else "FAIL"
        lines.append(
            f"{v.name:16} {v.n:>3} {v.baseline:>7.3f} {v.observed:>7.3f} "
            f"{v.delta_points:>+7.1f}p {v.evidence:>14}  {mark}: {v.reason}"
        )
    lines.append("")
    lines.append(
        "evidence = scenarios whose citations were checked against ingested text "
        "(DL-3).\nOnly the conflict slice has any. A pass on a slice at 0/n says the "
        "measurement\ndid not move, not that the measurement is right."
    )
    return "\n".join(lines)
