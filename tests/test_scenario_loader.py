from __future__ import annotations

import textwrap

import pytest

from eval.scenarios.loader import load_all, route_counts, unverified


def test_the_committed_set_loads() -> None:
    """A malformed scenario must break the build, not silently shrink the set."""
    assert load_all()


def test_every_route_is_exercised() -> None:
    """If a route has no scenarios, the agent can never be shown to get it
    wrong, and the metric for it is vacuous."""
    counts = route_counts()
    for route in ("answer", "clarify", "refuse", "escalate"):
        assert counts[route] > 0, f"no scenarios exercise route {route!r}"


def test_clarify_cases_are_outnumbered_by_answer_cases() -> None:
    """DL-5. If clarifying is the majority behaviour, an agent that always asks
    scores well while being unusable."""
    counts = route_counts()
    assert counts["answer"] > counts["clarify"]


def test_unverified_scenarios_are_tracked() -> None:
    """DL-3. Nothing is scoreable yet; this asserts the flag is honest rather
    than that the set is verified."""
    assert len(unverified()) == len(load_all())


def _write(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_duplicate_scenario_ids_are_rejected(tmp_path) -> None:
    body = """
        - scenario_id: dup-1
          slice: straightforward
          question: q
          employee_context: {state: OH}
          as_of_date: 2026-01-01
          expected_route: answer
          expected_authority: federal
          required_citations: ["29 CFR 825.200"]
        - scenario_id: dup-1
          slice: straightforward
          question: q2
          employee_context: {state: OH}
          as_of_date: 2026-01-01
          expected_route: answer
          expected_authority: federal
          required_citations: ["29 CFR 825.100"]
    """
    _write(tmp_path, "straightforward.yaml", body)
    with pytest.raises(ValueError, match="duplicate scenario_id"):
        load_all(tmp_path)


def test_slice_must_match_filename(tmp_path) -> None:
    """Catches a scenario moved between files without its label updated, which
    would skew the balance DL-4 fixes in advance."""
    body = """
        - scenario_id: mismatch-1
          slice: conflict
          question: q
          employee_context: {state: OH}
          as_of_date: 2026-01-01
          expected_route: answer
          expected_authority: federal
          required_citations: ["29 CFR 825.200"]
    """
    _write(tmp_path, "straightforward.yaml", body)
    with pytest.raises(ValueError, match="does not match file"):
        load_all(tmp_path)
