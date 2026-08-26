"""The state carried through the graph, and the records nodes write into it.

Two properties are worth reading before the code.

**The trace is append-only by construction, not by discipline.** It is annotated
with a reducer, so a node that returns `{"trace": [event]}` *adds* that event;
there is no way for a node to return a shorter trace than it received. This
matters because the trace is the demo's product rather than a debug artifact
(spec, "Observability"), and a node that overwrote it would destroy the thing
being shown while every test still passed. Making it structural removes the
question of whether each node remembered to append.

**Nodes never re-derive.** `triage` extracts jurisdiction and `as_of` once and
writes them to state; `retrieve` reads them. Two nodes independently parsing
"last year" out of a question is two chances to disagree, and the disagreement
would be invisible.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from domain import Authority, EmployeeContext, Jurisdiction, MissingFact, Route
from retrieval.store import SearchHit

# Which of the spec's five precedence rules decided a resolution. Recorded
# rather than inferred, because "right answer from the wrong authority is luck,
# not correctness" (spec, Metrics) and the same applies one level down: the right
# authority for the wrong reason is also luck.
PrecedenceRule = Literal[
    "statutory_floor",             # rule 1: the employee-favourable floor wins
    "policy_may_exceed",           # rule 2: handbook above statute controls
    "effective_dating",            # rule 3: only what was in force on the date
    "silence_is_not_permission",   # rule 4: a silent layer overrides nothing
    "concurrence_tie_break",       # rule 5: highest layer that independently compels
    "indeterminate",               # rule 5's corollary: several layers defensible
    "not_reached",                 # resolve did not run: clarify, refuse, escalate
]


Outcome = Literal["grants", "denies", "silent"]


@dataclass(frozen=True)
class LayerFinding:
    """What one authority layer had to say. Every layer is recorded, including
    the ones that lost and the ones that said nothing.

    `speaks_to_question` is the distinction rule 4 turns on. A layer that is
    silent is not a layer that permits, and collapsing the two is how "the
    handbook doesn't mention it, so it's fine" gets produced.

    `outcome` separates two things that are both "speaking": a layer can address
    the question and grant nothing. `conflict-005` is exactly that. Federal FMLA
    covers grandparents only as next of kin for military caregiver leave, so for
    ordinary care it speaks and denies. The answer is "no", which is an answer
    rather than a refusal, and the layer that determined it is controlling.

    `generosity_rank` is the one comparison a model has to make, because "more
    generous to the employee" is not always arithmetic. 1 is most generous, ties
    allowed. It ranks the PROVISIONS against each other; it never says which one
    controls. That decision is `agent/precedence.py`, in code.
    """

    layer: Authority
    speaks_to_question: bool
    outcome: Outcome = "silent"
    citation: str | None = None
    says: str = ""
    generosity_rank: int | None = None


@dataclass(frozen=True)
class Resolution:
    """Structured, not prose. A sentence saying "state law controls here" cannot
    be scored, diffed, or shown in a trace panel.

    `controlling` is None when the layers genuinely tie, and `acceptable` names
    the tie. Spec: where federal and state each independently compel the same
    outcome, neither is "the" controlling authority, and demanding one would
    score a defensible answer wrong.
    """

    controlling: Authority | None
    rule: PrecedenceRule
    considered: list[LayerFinding] = field(default_factory=list)
    acceptable: list[Authority] = field(default_factory=list)
    # Sources the answer must acknowledge without treating as controlling. An
    # employee who already read the handbook needs to know why the answer
    # differs from it, or the answer is useless to them.
    non_controlling_to_address: list[str] = field(default_factory=list)

    @property
    def defensible(self) -> list[Authority]:
        """Every authority this resolution treats as correct. One when it is
        determinate, several when it is not."""
        if self.acceptable:
            return list(self.acceptable)
        return [self.controlling] if self.controlling else []


@dataclass(frozen=True)
class VerificationResult:
    """`checks` is per-check rather than a single boolean so a failure says which
    property broke. Most of these are deterministic; see `agent/nodes/verify.py`."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TraceEvent:
    """One node's contribution to the visible reasoning trace.

    `summary` is written for a non-technical reader, because explaining an AI
    system to non-AI stakeholders is the job this project is auditioning for.
    `detail` carries the structured form for the trace panel and for Phase 7.
    """

    node: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """Total=False because nodes fill this in progressively. A node reads what
    earlier nodes wrote and writes only its own keys; LangGraph merges."""

    # Input.
    question: str
    employee_context: EmployeeContext
    as_of: date

    # triage. `rewritten_query` is what actually reaches the index: the raw
    # question is kept beside it so the two can be compared, which is what
    # `eval/baseline_retrieval.json` exists to check (DL-16).
    route: Route | None
    rewritten_query: str | None
    jurisdiction: Jurisdiction | None
    missing_fact: MissingFact | None

    # retrieve.
    retrieved: list[SearchHit]

    # resolve.
    resolution: Resolution | None

    # compose.
    answer: str | None
    citations: list[str]

    # verify.
    verification: VerificationResult | None

    # Append-only. See the module docstring: the reducer is the guarantee.
    trace: Annotated[list[TraceEvent], operator.add]


def initial_state(
    question: str,
    employee_context: EmployeeContext | None = None,
    as_of: date | None = None,
) -> AgentState:
    """`as_of` defaults to today only here, at the edge.

    Deliberately not defaulted inside the graph. A node that quietly substitutes
    today's date for a missing one answers a 2023 question with 2026 law and
    reports no error (DL-16). Making the default explicit and singular means
    there is one place to look when a point-in-time answer is wrong.
    """
    return AgentState(
        question=question,
        employee_context=employee_context or EmployeeContext(),
        as_of=as_of or date.today(),
        route=None,
        rewritten_query=None,
        jurisdiction=None,
        missing_fact=None,
        retrieved=[],
        resolution=None,
        answer=None,
        citations=[],
        verification=None,
        trace=[],
    )


def nodes_visited(state: AgentState) -> list[str]:
    return [event.node for event in state.get("trace", [])]
