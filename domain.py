"""The vocabulary shared by the agent and the harness that scores it.

These lived in `eval/scenarios/schema.py`, which made the agent import from the
thing that measures it. That dependency points the wrong way, and the obvious
alternative, a second copy of the same literals in `agent/`, is the failure this
project keeps rediscovering: two definitions that agree until one is edited.

So they sit here, below both. `eval` depends on this; `agent` depends on this;
neither depends on the other.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel

Route = Literal["answer", "clarify", "refuse", "escalate"]
Authority = Literal["federal", "state", "company"]
Jurisdiction = Literal["CA", "NY", "OH"]

# Facts an agent may legitimately need before it can answer. A `clarify` case
# must name which one is missing, so that asking the right question and asking
# merely to be safe can be told apart.
#
# weeks_worked_12mo exists because New York Paid Family Leave keys eligibility on
# weeks worked rather than months of service plus hours, and the first draft of
# the scenario set had a case asserting an NY outcome the schema could not
# express.
MissingFact = Literal[
    "state",
    "tenure_months",
    "hours_worked_12mo",
    "weeks_worked_12mo",
    "employer_size",
]


class EmployeeContext(BaseModel):
    """What the asker has volunteered. `None` means genuinely not supplied."""

    state: Jurisdiction | None = None
    tenure_months: int | None = None
    hours_worked_12mo: int | None = None
    weeks_worked_12mo: int | None = None
    employer_size: int | None = None


def missing_facts() -> tuple[str, ...]:
    return get_args(MissingFact)
