"""Scenario schema: the ground truth every metric is computed against.

The validators here are deliberately strict. A scenario that cannot be scored
unambiguously is worse than no scenario, because it produces a number that looks
like a measurement and is not. Each rule below closes off a way of writing an
unscoreable case.

Several of them exist because a review found the corresponding defect in the
first draft of the set. Those are marked. Prose intent is not enforcement: every
property the design depends on has to be a field a validator can check, or it
drifts silently. See DL-7.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Route, Authority, Jurisdiction, MissingFact and EmployeeContext describe the
# domain rather than the eval, and the agent needs them too. They live in
# `domain.py` so there is exactly one definition; re-exported here because the
# scenario files and every existing importer read them from this module.
from domain import (  # noqa: F401
    Authority,
    EmployeeContext,
    Jurisdiction,
    MissingFact,
    Route,
)

Slice = Literal[
    "straightforward",
    "ambiguous",
    "control",
    "conflict",
    "superseded",
    "out_of_scope",
    "adversarial",
]

class Scenario(BaseModel):
    scenario_id: str
    slice: Slice
    question: str
    employee_context: EmployeeContext
    as_of_date: date

    expected_route: Route

    # Exactly one of these is set on an `answer` scenario.
    #
    # expected_authority: one layer controls and it is determinate.
    # acceptable_authorities: the answer is determinate but the controlling
    #   layer is not, because jurisdiction was withheld and the layers agree on
    #   the outcome while differing on which one compels it. Scoring these
    #   against a single value would penalise a correct system for a fact it was
    #   never given. See spec, "The controlling authority can be indeterminate".
    expected_authority: Authority | None = None
    acceptable_authorities: list[Authority] = Field(default_factory=list)

    # Citations the answer must contain, and citations whose presence is a
    # failure: superseded provisions and other jurisdictions' law.
    required_citations: list[str] = Field(default_factory=list)
    forbidden_citations: list[str] = Field(default_factory=list)

    # Sources the answer must acknowledge without treating as controlling. A
    # conflict answer that never mentions the handbook leaves the reader unable
    # to reconcile it with what they already read.
    must_address: list[str] = Field(default_factory=list)

    # Required when expected_route is "clarify".
    missing_fact: MissingFact | None = None

    # The scenario this one is paired against. The design depends on pairs, and
    # in the first draft the pairings lived only in prose, where one of them was
    # wrong and nothing caught it. The loader verifies these resolve and are
    # reciprocal.
    pairs_with: str | None = None

    # DL-3: ground truth drafted from recall is not ground truth. A scenario is
    # not scored until its citations have been checked against ingested text.
    verified: bool = False

    notes: str = ""

    @model_validator(mode="after")
    def _check_route_consistency(self) -> Scenario:
        if self.expected_route == "answer":
            has_single = self.expected_authority is not None
            has_set = bool(self.acceptable_authorities)
            if not (has_single or has_set):
                raise ValueError(
                    "an 'answer' scenario must set expected_authority, or "
                    "acceptable_authorities where the controlling layer is indeterminate"
                )
            if has_single and has_set:
                raise ValueError(
                    "set expected_authority or acceptable_authorities, not both"
                )
            if has_set and len(self.acceptable_authorities) < 2:
                raise ValueError(
                    "acceptable_authorities is for genuinely indeterminate cases and "
                    "needs at least two entries; use expected_authority otherwise"
                )
            if not self.required_citations:
                raise ValueError("an 'answer' scenario must require at least one citation")

        if self.expected_route in ("refuse", "escalate"):
            if self.expected_authority is not None or self.acceptable_authorities:
                raise ValueError(
                    f"a '{self.expected_route}' scenario must not name a controlling "
                    "authority: neither route asserts an entitlement"
                )
            if self.required_citations:
                raise ValueError(
                    f"a '{self.expected_route}' scenario must not require citations"
                )

        if self.expected_route == "clarify":
            if self.missing_fact is None:
                raise ValueError(
                    "a 'clarify' scenario must name the missing fact, otherwise a correct "
                    "question and a merely cautious one cannot be told apart"
                )
            # Found by review: one scenario declared a missing fact that the
            # context supplied, which is unscoreable in both directions.
            supplied = getattr(self.employee_context, self.missing_fact)
            if supplied is not None:
                raise ValueError(
                    f"missing_fact is {self.missing_fact!r} but employee_context supplies "
                    f"it ({supplied!r}); the fact is not missing"
                )

        if self.expected_route != "clarify" and self.missing_fact is not None:
            raise ValueError("missing_fact is only meaningful on a 'clarify' scenario")

        overlap = set(self.required_citations) & set(self.forbidden_citations)
        if overlap:
            raise ValueError(f"citations both required and forbidden: {sorted(overlap)}")

        addressed_overlap = set(self.must_address) & set(self.required_citations)
        if addressed_overlap:
            raise ValueError(
                "must_address is for non-controlling sources; these are already "
                f"required: {sorted(addressed_overlap)}"
            )

        if self.pairs_with == self.scenario_id:
            raise ValueError("a scenario cannot pair with itself")

        return self
