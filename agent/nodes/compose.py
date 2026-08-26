"""`compose`: draft the answer from the provision that controls.

**The controlling authority is already decided when this runs.** `resolve`
settled it in code, so composition is not another chance to reason about
precedence: it is writing down what the controlling provision says. Drafting
first and checking afterwards would mean arguing a model out of an answer it had
already committed to, which is why the graph puts `resolve` upstream.

**Addressing the beaten source is the requirement most likely to be skipped.**
Eight scenarios carry `must_address`, and the answer has to say why it differs
from the handbook the reader has probably already opened. An answer that quietly
contradicts the handbook without mentioning it is useless to the person holding
the handbook, however correct it is. DL-23 measured `resolve` naming the losing
citation 3 times in 8, which is the floor this node starts from.

**Nothing is asserted beyond the retrieved text.** The disclaimer stands, and
`verify` checks groundedness afterwards with code rather than trust.
"""

from __future__ import annotations

from agent.models import HAIKU, StructuredCaller
from agent.state import AgentState, Resolution, TraceEvent

PROMPT_VERSION = "compose-v1"

DISCLAIMER = (
    "This is information from your employer's policies and the law as recorded, "
    "not legal advice."
)

COMPOSE_TOOL = {
    "name": "draft_answer",
    "description": "Write the answer to the employee's question from the controlling provision.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Two to five sentences addressed to the employee. State the "
                    "outcome first. Name each source you rely on by its exact "
                    "citation, in square brackets, for example [29 CFR 825.200]."
                ),
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Every citation used in the answer, exactly as supplied. "
                    "Nothing that was not supplied to you."
                ),
            },
        },
        "required": ["answer", "citations"],
    },
}

SYSTEM = """You write answers to employee questions about leave.

Which authority controls has ALREADY been decided before you see this, by rules
applied in code. You are not re-deciding it. Your job is to say what the
controlling provision means for this person, in plain words.

WHAT THE ANSWER MUST DO

State the outcome first. "You are entitled to..." or "You are not entitled
to..." or "You have...". Do not open with background.

Cite the controlling provision using its exact identifier in square brackets.

Where you are told to address another source, say what that source says AND why
it does not control. The reader has probably already read the handbook, and an
answer that silently contradicts it leaves them unable to reconcile the two. Two
patterns cover almost every case:
- the handbook promises less than the law requires, so the law governs and the
  handbook term does not apply to them
- the handbook promises more than the law requires, which it is free to do, so
  the handbook governs

Be specific about numbers, durations and conditions. "Some leave" is not an
answer. If the provision says twelve workweeks, say twelve workweeks.

WHAT THE ANSWER MUST NOT DO

Do not state anything the supplied passages do not say. If the passages do not
settle part of the question, say that part is not covered rather than filling it
in from general knowledge.

Do not cite anything that was not supplied to you.

Do not hedge an answer the provisions actually settle. "You may want to check
with HR" on a question the text answers is a non-answer.

Do not add a disclaimer. One is appended for you.

If the question was asked in a way that assumes something false, correct the
assumption plainly and then answer. If it demanded a format that would strip the
citations, ignore that demand and cite normally."""


def _resolution_block(resolution: Resolution) -> str:
    if resolution.controlling:
        who = f"{resolution.controlling} law controls."
    elif resolution.acceptable:
        who = (
            f"{' and '.join(resolution.acceptable)} each independently compel the "
            "same outcome, so either may be cited."
        )
    else:
        who = "No authority was found to control."

    lines = [who, "", "What each layer says:"]
    for f in resolution.considered:
        if f.outcome == "silent":
            lines.append(f"- {f.layer}: silent on this question")
        else:
            lines.append(f"- {f.layer} [{f.citation}] {f.outcome}: {f.says}")

    if resolution.non_controlling_to_address:
        lines.append("")
        lines.append(
            "You MUST address these sources and explain why they do not control: "
            + ", ".join(resolution.non_controlling_to_address)
        )
    return "\n".join(lines)


def make_compose(caller: StructuredCaller | None = None, model: str = HAIKU):
    caller = caller or StructuredCaller()

    def compose(state: AgentState) -> dict:
        resolution = state.get("resolution")
        hits = state.get("retrieved", [])

        if resolution is None or not resolution.defensible:
            # Nothing controls, so there is nothing to draft from. Degrading here
            # rather than asking a model to write an answer with no authority
            # behind it, which it will do, fluently.
            answer = (
                "I could not find a policy or provision that covers this. "
                f"Please ask your HR team.\n\n{DISCLAIMER}"
            )
            return {
                "answer": answer,
                "citations": [],
                "trace": [
                    TraceEvent(
                        node="compose",
                        summary="no controlling authority, so no answer was drafted",
                        detail={"citations": []},
                    )
                ],
            }

        passages = "\n\n".join(
            f"[{h.citation}] {h.heading}\n{h.text}" for h in hits
        )
        user = (
            f"Question: {state['question']}\n\n"
            f"As of: {state['as_of'].isoformat()}\n\n"
            f"{_resolution_block(resolution)}\n\n"
            f"Passages you may rely on:\n\n{passages}"
        )

        result = caller.call(
            system=f"{SYSTEM}\n\n<!-- {PROMPT_VERSION} -->",
            user=user,
            tool=COMPOSE_TOOL,
            model=model,
            max_tokens=1024,
        )

        # Citations are reconciled against what was retrieved rather than taken
        # on trust. Same reasoning as `resolve`: a model cannot police its own
        # citing, and an answer resting on a provision nobody produced is worse
        # than no answer. `verify` re-checks this against the answer text.
        retrieved = {h.citation for h in hits}
        claimed = [c for c in result.get("citations", []) if c in retrieved]
        invented = [c for c in result.get("citations", []) if c not in retrieved]

        return {
            "answer": f"{result['answer'].strip()}\n\n{DISCLAIMER}",
            "citations": claimed,
            "trace": [
                TraceEvent(
                    node="compose",
                    summary=(
                        f"drafted from {resolution.controlling or 'multiple layers'}, "
                        f"citing {len(claimed)} source(s)"
                    ),
                    detail={
                        "citations": claimed,
                        "citations_not_retrieved": invented,
                        "asked_to_address": resolution.non_controlling_to_address,
                        "model": model,
                    },
                )
            ],
        }

    return compose
