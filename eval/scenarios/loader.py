"""Load and validate the scenario set.

Loading is strict on purpose. A malformed scenario that silently fails to load
shrinks the eval set without changing any visible number, which is the worst
possible failure: the metrics still print, they just mean less than they say.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from eval.scenarios.schema import Scenario

SCENARIO_DIR = Path(__file__).resolve().parent


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

    return scenarios


def slice_counts(scenarios: list[Scenario] | None = None) -> Counter[str]:
    return Counter(s.slice for s in (scenarios or load_all()))


def route_counts(scenarios: list[Scenario] | None = None) -> Counter[str]:
    return Counter(s.expected_route for s in (scenarios or load_all()))


def unverified(scenarios: list[Scenario] | None = None) -> list[Scenario]:
    """DL-3: scenarios whose ground truth has not been checked against the
    ingested corpus. These must not be scored."""
    return [s for s in (scenarios or load_all()) if not s.verified]
