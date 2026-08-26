"""`refuse` and `escalate`: the two endings that are not answers.

Neither uses a model, for the same reason `clarify` does not. Triage has already
made the decision and written a sentence explaining it; restating that through a
second model would add cost and a way for the explanation to drift from the
decision it explains.

**They are different endings and the difference is the subject matter, not
whether a person gets involved** (spec, "Refuse and escalate"). Collapsing them
was the defect DL-7 found riding on eighteen scenarios: the first draft defined
refuse as "nothing covers this, here is who to ask", which is also a description
of escalate.

Neither asserts an entitlement, so neither carries a citation.
"""

from __future__ import annotations

from agent.nodes.compose import DISCLAIMER
from agent.state import AgentState, TraceEvent

# Nothing in the corpus bears on the question. Says so, and points somewhere
# better, without making any claim about what the person is owed.
REFUSE_TEXT = (
    "That is not something the leave and time-off policies cover, so I do not "
    "have a reliable answer for you. Your HR team will be able to help."
)

# The corpus does bear on it, but a correct response needs human judgment:
# a legal conclusion, a live dispute, distress, or another person's situation.
ESCALATE_TEXT = (
    "This needs a person rather than a policy lookup. I can see which rules bear "
    "on it, but reaching a conclusion here would be a judgment I should not make "
    "on your behalf. Please raise it with your HR team, and say it is time "
    "sensitive if it is."
)


def _reason_from_triage(state: AgentState) -> str:
    return next(
        (e.summary for e in reversed(state.get("trace", [])) if e.node == "triage"),
        "",
    )


def _ending(state: AgentState, node: str, text: str) -> dict:
    because = _reason_from_triage(state)
    body = f"{text}\n\n{because}".strip() if because else text
    return {
        "answer": f"{body}\n\n{DISCLAIMER}",
        "citations": [],
        "trace": [
            TraceEvent(
                node=node,
                summary=because or f"{node}d",
                detail={"route": node, "asserts_entitlement": False},
            )
        ],
    }


def refuse(state: AgentState) -> dict:
    return _ending(state, "refuse", REFUSE_TEXT)


def escalate(state: AgentState) -> dict:
    return _ending(state, "escalate", ESCALATE_TEXT)
