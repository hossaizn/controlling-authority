"""Mutation testing: break the source on purpose, and check the suite notices.

**This is the project's primary review technique, and it is committed because it
kept finding things nothing else did.** Every review that ran it found real
defects: eval scorers that could all be mutated to return `True` with 523 tests
green (DL-26), a figures check with three independent holes (DL-35), a tracing
export where 19 of 25 mutations survived (Phase 7 review), a rate limiter whose
eviction could delete live keys (Phase 8 review).

It also caught a subtler failure. After the first round of fixes for the scorer
defects, **nine of fifteen mutations still survived**, because the new tests
asserted that a value was *present* rather than what it *was*. That is DL-10's
rule arriving by a different road: a test that pins a relationship instead of a
value cannot tell a correct implementation from a broken one.

Three reviews each found a module with no coverage at all, and I could not see
those holes from inside the code. Running this reactively three times found the
instances; committing it is the attempt to fix the class, which is DL-26's
"fixing an instance of a bug is not fixing the bug".

    uv run python -m eval.mutation                 # every mutation
    uv run python -m eval.mutation verify          # only matching ones
    uv run python -m eval.mutation --list

**A mutation whose target no longer matches is an error, not a skip.** DL-12:
`str.replace()` returns the string unchanged when it finds nothing, so a stale
mutation fails open, reporting "caught" while never having been applied. Those
are reported separately and make the run exit non-zero, because the honest
reading of a stale mutation is that the guarantee it encoded is now unverified.

**If the process is killed, the source stays mutated.** The restore runs in a
`finally`, which a SIGKILL does not reach. A full sweep takes upwards of ten
minutes, so this is reached by any wrapper that imposes a shorter timeout, and
it happened once during Phase 10.

The recovery is `git checkout -- <file>`, and the reason it is merely annoying
rather than dangerous is the baseline check below: the next run refuses to
start against a red suite instead of reporting 146 false "caught" verdicts.
Run this in the background rather than under a timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = Path(__file__).resolve().parent / "mutations.json"

# A mutation is caught the moment any test fails, so there is no reason to run
# the rest of the suite. Turns a ~16 minute sweep into a few minutes.
PYTEST = ["uv", "run", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"]


@dataclass(frozen=True)
class Mutation:
    label: str
    file: str
    old: str
    new: str
    origin: str = ""

    @property
    def path(self) -> Path:
        return ROOT / self.file


def load() -> list[Mutation]:
    return [Mutation(**m) for m in json.loads(CATALOGUE.read_text())]


def suite_passes() -> bool:
    result = subprocess.run(PYTEST, cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


def apply_and_test(mutation: Mutation) -> str:
    """Returns 'caught', 'SURVIVED', or 'STALE'.

    The original text is restored in a `finally` and the restore is verified.
    A harness that leaves a mutated file behind on failure corrupts every
    subsequent measurement in the session, and the corruption looks like a
    real regression.
    """
    path = mutation.path
    if not path.exists():
        return "STALE"

    original = path.read_text()
    if mutation.old not in original:
        return "STALE"

    mutated = original.replace(mutation.old, mutation.new, 1)
    if mutated == original:
        # Target present but replacement is a no-op: the mutation does nothing,
        # so a green suite proves nothing about it.
        return "STALE"

    try:
        path.write_text(mutated)
        return "SURVIVED" if suite_passes() else "caught"
    finally:
        path.write_text(original)
        if path.read_text() != original:
            raise SystemExit(f"FAILED TO RESTORE {mutation.file} - fix before rerunning")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    mutations = load()

    if "--list" in sys.argv:
        for m in mutations:
            print(f"  {m.file:28} {m.label}")
        print(f"\n{len(mutations)} mutations")
        return 0

    if args:
        needle = args[0].lower()
        mutations = [
            m for m in mutations
            if needle in m.file.lower() or needle in m.label.lower()
        ]
        if not mutations:
            print(f"no mutation matches {needle!r}")
            return 1

    print("baseline: ", end="", flush=True)
    if not suite_passes():
        print("SUITE IS ALREADY RED. Fix that first; every mutation would "
              "report 'caught' for the wrong reason.")
        return 1
    print("green\n")

    survivors: list[Mutation] = []
    stale: list[Mutation] = []

    for i, m in enumerate(mutations, 1):
        verdict = apply_and_test(m)
        print(f"  {i:3}/{len(mutations)}  {verdict:9} {m.file:28} {m.label}",
              flush=True)
        if verdict == "SURVIVED":
            survivors.append(m)
        elif verdict == "STALE":
            stale.append(m)

    caught = len(mutations) - len(survivors) - len(stale)
    print(f"\n{caught}/{len(mutations)} caught, "
          f"{len(survivors)} survived, {len(stale)} stale")

    if survivors:
        print("\nSURVIVED - the suite cannot tell these from correct code:")
        for m in survivors:
            print(f"  {m.file:28} {m.label}")

    if stale:
        print("\nSTALE - target no longer in the source, so the guarantee "
              "these encoded is now unverified:")
        for m in stale:
            print(f"  {m.file:28} {m.label}   ({m.origin})")

    return 1 if survivors or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
