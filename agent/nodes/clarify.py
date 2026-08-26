"""`clarify`: ask for the one fact that changes the answer.

**No model call, deliberately.** Triage has already decided the route and named
the missing fact. Turning `tenure_months` into "How long have you worked here?"
is a five-way lookup, not reasoning. A model call here would add latency, cost
and a hallucination surface to a mapping that a dict does correctly every time.

Same reasoning that makes `verify` mostly deterministic: where the work is a
function, write the function. It also means the clarifying question cannot drift
away from the fact it is supposed to be asking about, which is a failure a model
could produce and nothing downstream would catch, because `missing_fact` would
still be right in the trace while the sentence asked for something else.

**Whether to ask at all is scored, not decided here.** That is triage's call and
`eval/run_routes.py` measures both directions of it. The control slice exists to
punish asking when the answer would not change, and an agent that always asks is
trivially safe and unusable (DL-5).
"""

from __future__ import annotations

from agent.state import AgentState, TraceEvent
from domain import MissingFact, missing_facts

# Written to be read by the person who asked, not by a developer. Each one names
# the fact plainly enough that the answer can be typed in a few words.
QUESTIONS: dict[MissingFact, str] = {
    "state": "Which state do you work in?",
    "tenure_months": "How long have you worked for the company?",
    "hours_worked_12mo": (
        "Roughly how many hours have you worked in the last 12 months?"
    ),
    "weeks_worked_12mo": (
        "Roughly how many weeks have you worked in the last 12 months?"
    ),
    "employer_size": "Roughly how many people does your employer have?",
}

# A fact with no question would emit an empty clarification, which reads as a
# system failure to the person asking. Checked at import so it cannot ship.
assert set(QUESTIONS) == set(missing_facts()), (
    f"every MissingFact needs a question: missing {set(missing_facts()) - set(QUESTIONS)}"
)


def clarify(state: AgentState) -> dict:
    fact = state.get("missing_fact")
    if fact is None:
        # Unreachable through triage, which downgrades a factless clarify to
        # answer. Raising rather than emitting a vague "could you tell me more":
        # a question that names nothing cannot be answered and wastes the turn.
        raise ValueError("clarify reached with no missing_fact; nothing to ask about")

    question = QUESTIONS[fact]
    # Triage's own sentence explains why the fact matters. It is already written
    # for the asker and already in the trace, so reusing it costs nothing and
    # keeps the explanation consistent with the decision that produced it.
    because = next(
        (e.summary for e in reversed(state.get("trace", [])) if e.node == "triage"),
        "",
    )
    answer = f"{question}\n\n{because}".strip() if because else question

    return {
        "answer": answer,
        "citations": [],
        "trace": [
            TraceEvent(
                node="clarify",
                summary=f"asked for {fact}",
                detail={"missing_fact": fact, "question": question},
            )
        ],
    }
