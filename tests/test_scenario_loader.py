from __future__ import annotations

import textwrap

import pytest

from eval.scenarios.loader import handbook_policy_ids, load_all, route_counts, unverified


def test_the_committed_set_loads() -> None:
    """A malformed scenario must break the build, not silently shrink the set."""
    assert load_all()


def test_every_route_is_exercised() -> None:
    """If a route has no scenarios, the agent can never be shown to get it
    wrong, and the metric for it is vacuous."""
    counts = route_counts()
    for route in ("answer", "clarify", "refuse", "escalate"):
        assert counts[route] > 0, f"no scenarios exercise route {route!r}"


def test_unverified_scenarios_are_a_subset_not_the_whole_set() -> None:
    """DL-3 tracking must survive verification actually happening.

    The original assertion here was `len(unverified()) == len(load_all())`,
    which would have failed the moment Phase 3 verified its first scenario: a
    test that breaks when the project succeeds. What matters is that the flag
    is tracked and that nothing claims verification it has not earned.
    """
    all_scenarios = load_all()
    pending = unverified(all_scenarios)
    assert len(pending) <= len(all_scenarios)
    assert all(s.verified is False for s in pending)


def test_every_handbook_citation_resolves() -> None:
    """Citing a policy that does not exist would score a correct answer wrong.

    load_all() enforces this; this test states it as a property so the reason is
    visible rather than buried in the loader.
    """
    known = handbook_policy_ids()
    assert "LEAVE-004-v1" in known and "LEAVE-004-v2" in known
    # A versioned policy must not be citable without its version, or a
    # supersession scenario could cite it ambiguously and pass either way.
    assert "LEAVE-004" not in known


def test_pairings_are_reciprocal_and_meaningful() -> None:
    """The design leans on paired cases; in the first draft the pairings lived
    only in prose and one of them was simply false."""
    by_id = {s.scenario_id: s for s in load_all()}
    paired = [s for s in by_id.values() if s.pairs_with]
    assert paired, "no pairings declared"
    for s in paired:
        partner = by_id[s.pairs_with]
        assert partner.pairs_with == s.scenario_id
        # A pair must contrast on something. Identical route and authority on
        # both sides means the pair demonstrates nothing.
        differs = (
            s.expected_route != partner.expected_route
            or s.expected_authority != partner.expected_authority
            or s.employee_context != partner.employee_context
            or s.as_of_date != partner.as_of_date
            or s.required_citations != partner.required_citations
            # Two clarify cases can be a valid pair by disagreeing on which
            # prong is missing, which is the point of ambiguous-006/007.
            or s.missing_fact != partner.missing_fact
        )
        assert differs, f"{s.scenario_id} and {partner.scenario_id} contrast on nothing"


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


def test_dangling_pairing_is_rejected(tmp_path) -> None:
    body = """
        - scenario_id: lonely-1
          slice: straightforward
          question: q
          employee_context: {state: OH}
          as_of_date: 2026-01-01
          expected_route: answer
          expected_authority: federal
          required_citations: ["29 CFR 825.200"]
          pairs_with: does-not-exist
    """
    _write(tmp_path, "straightforward.yaml", body)
    with pytest.raises(ValueError, match="does not exist"):
        load_all(tmp_path)


def test_one_way_pairing_is_rejected(tmp_path) -> None:
    """The exact defect review found: A claims B as its pair, B knows nothing
    about A."""
    body = """
        - scenario_id: a-1
          slice: straightforward
          question: q
          employee_context: {state: OH}
          as_of_date: 2026-01-01
          expected_route: answer
          expected_authority: federal
          required_citations: ["29 CFR 825.200"]
          pairs_with: b-1
        - scenario_id: b-1
          slice: straightforward
          question: q
          employee_context: {state: CA}
          as_of_date: 2026-01-01
          expected_route: answer
          expected_authority: state
          required_citations: ["Cal. Gov. Code 12945.2"]
    """
    _write(tmp_path, "straightforward.yaml", body)
    with pytest.raises(ValueError, match="pairs with"):
        load_all(tmp_path)
