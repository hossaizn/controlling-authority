"""Load and validate the scenario set.

Loading is strict on purpose. A malformed scenario that silently fails to load
shrinks the eval set without changing any visible number, which is the worst
possible failure: the metrics still print, they just mean less than they say.

The cross-scenario and cross-corpus checks below exist because a review of the
first draft found a pairing that was asserted in prose and false in fact. Any
property the design relies on has to be checkable, or it drifts. See DL-7.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

from eval.scenarios.schema import Scenario

SCENARIO_DIR = Path(__file__).resolve().parent
HANDBOOK_DIR = SCENARIO_DIR.parent.parent / "corpus" / "handbook"

# Handbook citations look like LEAVE-003 or LEAVE-004-v2. Statutory citations
# are anything else and are verified against ingested text in Phase 3 (DL-3),
# not here.
HANDBOOK_CITATION = re.compile(r"^LEAVE-\d{3}(?:-v(\d+))?$")


@lru_cache(maxsize=1)
def handbook_policy_ids() -> frozenset[str]:
    """Every citable identifier the handbook actually defines.

    A policy with versions is citable only by version, so that a scenario
    turning on supersession cannot cite the policy ambiguously.
    """
    ids: set[str] = set()
    for path in HANDBOOK_DIR.glob("LEAVE-*.md"):
        text = path.read_text()
        policy_id = re.search(r"^policy_id:\s*(\S+)", text, re.M)
        if not policy_id:
            continue
        version = re.search(r"^version:\s*(\d+)", text, re.M)
        ids.add(f"{policy_id.group(1)}-v{version.group(1)}" if version else policy_id.group(1))
    return frozenset(ids)


def load_all(directory: Path | None = None) -> list[Scenario]:
    """Load every scenario YAML in the directory, validating each one.

    Raises on the first invalid scenario rather than skipping it.
    """
    directory = directory or SCENARIO_DIR
    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}

    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or []
        if not isinstance(raw, list):
            raise ValueError(f"{path.name}: expected a list of scenarios")

        for entry in raw:
            scenario = Scenario(**entry)

            if scenario.scenario_id in seen:
                raise ValueError(
                    f"duplicate scenario_id {scenario.scenario_id!r} in {path.name}, "
                    f"already defined in {seen[scenario.scenario_id].name}"
                )
            seen[scenario.scenario_id] = path

            # The filename is the slice. A mismatch means a scenario was moved
            # without its label being updated, which would quietly skew the
            # balance that DL-4 fixes in advance.
            if scenario.slice != path.stem:
                raise ValueError(
                    f"{scenario.scenario_id}: slice {scenario.slice!r} does not match "
                    f"file {path.name}"
                )

            scenarios.append(scenario)

    _check_pairs(scenarios)
    if directory == SCENARIO_DIR:
        _check_handbook_citations(scenarios)
    return scenarios


def _check_pairs(scenarios: list[Scenario]) -> None:
    """Pairings must resolve and be reciprocal.

    A one-way pairing is how the first draft ended up with a scenario naming a
    partner that had nothing to do with it.
    """
    by_id = {s.scenario_id: s for s in scenarios}
    for s in scenarios:
        if s.pairs_with is None:
            continue
        partner = by_id.get(s.pairs_with)
        if partner is None:
            raise ValueError(f"{s.scenario_id}: pairs_with {s.pairs_with!r} does not exist")
        if partner.pairs_with != s.scenario_id:
            raise ValueError(
                f"{s.scenario_id} pairs with {partner.scenario_id}, but "
                f"{partner.scenario_id} pairs with {partner.pairs_with!r}"
            )


def _check_handbook_citations(scenarios: list[Scenario]) -> None:
    """Every handbook citation must resolve to a policy that exists.

    Statutory citations are out of scope here; they are checked against ingested
    text in Phase 3.
    """
    known = handbook_policy_ids()
    unresolved: list[str] = []
    malformed: list[str] = []
    for s in scenarios:
        cites = s.required_citations + s.forbidden_citations + s.must_address
        for cite in cites:
            # A citation carrying quotes or backslashes is a YAML authoring
            # error, not a citation. Six of these slipped through the first
            # version of this check because it only inspected strings that
            # already looked well formed, so a malformed one skipped validation
            # rather than failing it. A guard that ignores what it cannot
            # recognise is not a guard.
            # Double quotes and backslashes are authoring errors. Apostrophes
            # are not: "N.Y. Workers' Comp. Law 204" legitimately contains one.
            if '"' in cite or "\\" in cite:
                malformed.append(f"{s.scenario_id} -> {cite!r}")
                continue
            if HANDBOOK_CITATION.match(cite) and cite not in known:
                unresolved.append(f"{s.scenario_id} -> {cite}")
    if malformed:
        raise ValueError(f"malformed citation strings: {malformed}")
    if unresolved:
        raise ValueError(f"handbook citations that do not resolve: {unresolved}")


def slice_counts(scenarios: list[Scenario] | None = None) -> Counter[str]:
    return Counter(s.slice for s in (scenarios or load_all()))


def route_counts(scenarios: list[Scenario] | None = None) -> Counter[str]:
    return Counter(s.expected_route for s in (scenarios or load_all()))


def unverified(scenarios: list[Scenario] | None = None) -> list[Scenario]:
    """DL-3: scenarios whose ground truth has not been checked against the
    ingested corpus. These must not be scored."""
    return [s for s in (scenarios or load_all()) if not s.verified]
