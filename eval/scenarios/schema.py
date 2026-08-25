"""Scenario schema: the ground truth every metric is computed against.

The validators here are deliberately strict. A scenario that cannot be scored
unambiguously is worse than no scenario, because it produces a number that looks
like a measurement and is not one. Each rule below closes off a way of writing
an unscoreable case.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Route = Literal["answer", "clarify", "refuse", "escalate"]
Authority = Literal["federal", "state", "company"]
Jurisdiction = Literal["CA", "NY", "OH"]
Slice = Literal[
    "straightforward",
    "ambiguous",
    "control",
    "conflict",
    "superseded",
    "out_of_scope",
    "adversarial",
]

# Facts an agent may legitimately need before it can answer. A `clarify`
# scenario must name which one is missing, so that asking the right question and
# asking merely to be safe can be told apart.
MissingFact = Literal["state", "tenure_months", "hours_worked_12mo", "employer_size"]


class EmployeeContext(BaseModel):
    """What the asker has volunteered. `None` means genuinely not supplied."""

    state: Jurisdiction | None = None
    tenure_months: int | None = None
    hours_worked_12mo: int | None = None
    employer_size: int | None = None


class Scenario(BaseModel):
    scenario_id: str
    slice: Slice
    question: str
    employee_context: EmployeeContext
    as_of_date: date

    expected_route: Route
    expected_authority: Authority | None = None

    # Citations the answer must contain, and citations whose presence is a
    # failure: superseded provisions and other jurisdictions' law.
    required_citations: list[str] = Field(default_factory=list)
    forbidden_citations: list[str] = Field(default_factory=list)

    # Required when expected_route is "clarify".
    missing_fact: MissingFact | None = None

    # DL-3: ground truth drafted from recall is not ground truth. A scenario is
    # not scored until its citations have been checked against ingested text.
    verified: bool = False

    notes: str = ""

    @model_validator(mode="after")
    def _check_route_consistency(self) -> Scenario:
        if self.expected_route == "answer":
            if self.expected_authority is None:
                raise ValueError("an 'answer' scenario must name the controlling authority")
            if not self.required_citations:
                raise ValueError("an 'answer' scenario must require at least one citation")

        if self.expected_route == "refuse" and self.expected_authority is not None:
            raise ValueError(
                "a 'refuse' scenario must not name a controlling authority: "
                "refusing means no source governs"
            )

        if self.expected_route == "clarify" and self.missing_fact is None:
            raise ValueError(
                "a 'clarify' scenario must name the missing fact, otherwise a correct "
                "question and a merely cautious one cannot be told apart"
            )

        overlap = set(self.required_citations) & set(self.forbidden_citations)
        if overlap:
            raise ValueError(f"citations both required and forbidden: {sorted(overlap)}")

        return self
