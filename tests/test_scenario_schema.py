from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from eval.scenarios.schema import EmployeeContext, Scenario


def _minimal(**overrides):
    base = dict(
        scenario_id="s1",
        slice="straightforward",
        question="How much FMLA leave am I entitled to?",
        employee_context=EmployeeContext(state="OH", tenure_months=24, hours_worked_12mo=2000),
        as_of_date=date(2026, 1, 1),
        expected_route="answer",
        expected_authority="federal",
        required_citations=["29 CFR 825.200"],
    )
    base.update(overrides)
    return base


def test_expected_route_is_required() -> None:
    args = _minimal()
    del args["expected_route"]
    with pytest.raises(ValidationError):
        Scenario(**args)


def test_expected_route_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        Scenario(**_minimal(expected_route="maybe"))


def test_answer_route_requires_an_authority() -> None:
    """An 'answer' with no controlling authority is an unscoreable scenario."""
    with pytest.raises(ValidationError):
        Scenario(**_minimal(expected_route="answer", expected_authority=None))


def test_answer_route_requires_at_least_one_citation() -> None:
    with pytest.raises(ValidationError):
        Scenario(**_minimal(required_citations=[]))


def test_refuse_route_must_not_carry_an_authority() -> None:
    """Refusing means no source governs. Naming one contradicts the route."""
    with pytest.raises(ValidationError):
        Scenario(
            **_minimal(
                slice="out_of_scope",
                expected_route="refuse",
                expected_authority="federal",
                required_citations=[],
            )
        )


def test_clarify_route_names_the_missing_fact() -> None:
    """A clarify scenario must say which fact is missing, or the agent's
    question cannot be scored as correct or merely cautious."""
    with pytest.raises(ValidationError):
        Scenario(
            **_minimal(
                slice="ambiguous",
                expected_route="clarify",
                expected_authority=None,
                required_citations=[],
            )
        )


def test_citation_cannot_be_both_required_and_forbidden() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            **_minimal(
                required_citations=["29 CFR 825.200"],
                forbidden_citations=["29 CFR 825.200"],
            )
        )


def test_unverified_is_the_default() -> None:
    """DL-3: ground truth is unverified until checked against ingested text."""
    assert Scenario(**_minimal()).verified is False


def test_valid_scenario_round_trips() -> None:
    s = Scenario(**_minimal())
    assert s.expected_route == "answer"
    assert s.employee_context.state == "OH"
