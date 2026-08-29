"""`resolve`: read what each authority layer says, then apply precedence in code.

The split is the point. The model is asked only what the retrieved text says and
which provision is more generous to the employee. It is never asked which layer
controls: that is `agent/precedence.py`, where the rule that fired is recorded,
the same input always gives the same output, and rule 5 cannot be argued out of
by a persuasively worded question.

Passages are presented **grouped by layer and labelled**, because the layer a
provision belongs to is the whole basis of the decision and asking a model to
infer it from a citation string is inviting an error that precedence cannot
detect. It is metadata we already have.

**Absence records read as silence, and that is deliberate.** A retrieved record
saying Ohio has no family-leave provision is evidence that the layer is silent,
which is different from retrieval having failed. Rule 4 then keeps that silence
from overriding a layer that does speak.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from agent.models import DEFAULT_TEMPERATURE, HAIKU, StructuredCaller
from agent.precedence import PrecedenceError, resolve_precedence
from agent.state import AgentState, LayerFinding, Resolution, TraceEvent
from domain import Authority

PROMPT_VERSION = "resolve-v4"

LAYERS: tuple[Authority, ...] = ("federal", "state", "company")

# The tool requires a rank on every finding because models fill required fields
# more reliably than optional ones. 0 means "not applicable", which is what a
# silent layer gets, and the node maps it to None before precedence sees it.
NOT_APPLICABLE = 0

READ_TOOL = {
    "name": "read_authorities",
    "description": (
        "Record what each authority layer says about the question, and how the "
        "provisions compare in generosity to the employee."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": "Exactly one entry per layer: federal, state, company.",
                "items": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "string", "enum": list(LAYERS)},
                        "outcome": {
                            "type": "string",
                            "enum": ["grants", "denies", "silent"],
                            "description": (
                                "grants: this layer provides the entitlement asked "
                                "about. denies: it addresses the question and does "
                                "not provide it. silent: it does not address the "
                                "question at all."
                            ),
                        },
                        "citation": {
                            "type": "string",
                            "description": (
                                "The bracketed identifier only, for example "
                                "'29 CFR 825.200'. Not the passage text, not the "
                                "heading. Empty string when the layer is silent."
                            ),
                        },
                        "says": {
                            "type": "string",
                            "description": "One sentence, what this layer provides.",
                        },
                        "generosity_rank": {
                            "type": "integer",
                            "description": (
                                "1 is most generous to the employee. Equal ranks "
                                "mean the layers provide the same thing; rank them "
                                "differently wherever they differ. Use 0 when the "
                                "layer is silent. A layer that denies must never "
                                "rank better than one that grants."
                            ),
                        },
                    },
                    "required": ["layer", "outcome", "citation", "says", "generosity_rank"],
                },
            }
        },
        "required": ["findings"],
    },
}

SYSTEM = """You read employment leave provisions and report what each one says.

You are NOT deciding which authority wins. That decision is made by code from
what you report, and it follows rules you do not need to know. Reporting the
evidence accurately is the entire job; guessing the outcome corrupts it.

For each of the three layers, federal, state and company, report:

OUTCOME
- grants: this layer provides the entitlement being asked about.
- denies: this layer addresses the question and does not provide it. A rule that
  defines who or what is covered, where the asker falls outside it, denies.
- silent: this layer does not address the question at all.

"denies" and "silent" are different and the difference matters. A statute that
lists covered family members and omits grandparents DENIES leave for a
grandparent. A statute that never mentions family leave is SILENT. If a passage
explicitly records that a state has no provision on the topic, that layer is
silent, and you know it rather than merely failing to find it.

If no passage from a layer was supplied, that layer is silent.

CITATION
Every passage is preceded by its citation in square brackets. Return ONLY that
bracketed identifier, with nothing else: no brackets, no heading, no passage
text. For example, given

    [29 CFR 825.200] § 825.200 Amount of leave. (a) An eligible employee...

the citation is exactly:

    29 CFR 825.200

Return one citation, not several. Never return a citation that was not supplied
to you. Use an empty string for a silent layer.

GENEROSITY RANK
Rank the layers by how favourable they are TO THE EMPLOYEE. 1 is most generous.

- More leave, more pay, broader coverage, or an easier eligibility test is more
  generous. A lower service requirement is more generous, not less.
- A layer that denies must never rank better than one that grants.
- Silent layers get 0.

Ties and differences are equally important to get right, and both are errors in
opposite directions.

- Tie them ONLY where they provide the same thing. A policy that restates a
  statutory minimum ties with it, and breaking that tie arbitrarily is wrong.
- Do NOT tie them where they differ. More days, more weeks, paid rather than
  unpaid, a longer period of coverage, more qualifying reasons, or a lower
  service requirement all mean one is more generous than the other. Ten paid
  days against a five-day statutory minimum is not a tie.

Read the two provisions against each other on the specific point and say which
gives the employee more. Only call it equal when neither does.

Compare the provisions on the point the question actually turns on. If the
question is about eligibility, compare the eligibility tests, not the amount of
leave each one grants."""


# How many passages of each layer reach the prompt. `None` is every passage,
# which is what this node did before the question was asked at all.
#
# **Per layer rather than overall**, measured in DL-38: federal is 54% of the
# passage tokens at a mean of 6.11 passages, while state is 2.28 and company
# 1.74. A global cut would thin the layers that are already thin, and precedence
# needs a finding from each layer to compare anything.
DEFAULT_PASSAGE_CAP: int | None = None


def cap_per_layer(hits: list, cap: int | None) -> list:
    """The top `cap` passages of each layer, in retrieval order.

    Retrieval order is preserved rather than regrouped, because rank within a
    layer is the signal being trusted: DL-38 measured that 84% of the evidence
    `resolve` cites is the top passage of its layer and 96.6% is within the top
    three.
    """
    if cap is None:
        return list(hits)
    kept: list = []
    seen: dict[str, int] = defaultdict(int)
    for hit in hits:
        if seen[hit.authority_layer] < cap:
            kept.append(hit)
            seen[hit.authority_layer] += 1
    return kept


def _passages_by_layer(state: AgentState) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for hit in state.get("retrieved", []):
        grouped[hit.authority_layer].append(hit)
    return grouped


def _user_message(state: AgentState) -> str:
    ctx = state["employee_context"]
    supplied = {k: v for k, v in ctx.model_dump().items() if v is not None}
    known = "\n".join(f"- {k}: {v}" for k, v in supplied.items()) or "- nothing supplied"

    grouped = _passages_by_layer(state)
    blocks = []
    for layer in LAYERS:
        hits = grouped.get(layer, [])
        if not hits:
            blocks.append(f"### {layer.upper()}\n(no passages retrieved for this layer)")
            continue
        body = "\n\n".join(
            f"[{h.citation}] {h.heading}\n{h.text}"
            + ("\n(this is a record that the law is silent on this topic)"
               if h.content_status == "absent" else "")
            for h in hits
        )
        blocks.append(f"### {layer.upper()}\n{body}")

    return (
        f"Question: {state['question']}\n\n"
        f"What is known about the employee:\n{known}\n\n"
        f"As of date: {state['as_of'].isoformat()}\n\n"
        "Retrieved passages, grouped by authority layer:\n\n" + "\n\n".join(blocks)
    )


def resolve_citation(raw: str, valid: set[str]) -> str | None:
    """Map whatever the model wrote into a citation that was actually retrieved.

    **Necessary because the model returned whole passages here.** Told to copy
    the citation "exactly as it appears in the passages", and given passages
    rendered as `[citation] heading / body`, it copied the entire block. That
    silently emptied `non_controlling_to_address` and would have broken every
    citation check in `verify`, while the resolution still looked correct.

    Doing it in code rather than only in the prompt is the point: this also
    rejects a citation that was never retrieved, which no amount of prompting
    reliably prevents and which a model cannot be trusted to police in itself.

    **Longest match wins.** `Cal. Gov. Code 12945` is a prefix of
    `Cal. Gov. Code 12945.2` and both are in this corpus. DL-12 records the same
    trap producing the right citation attached to the wrong statute.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if text in valid:
        return text
    matches = [c for c in valid if c in text]
    return max(matches, key=len) if matches else None


def _to_findings(raw: list[dict], valid_citations: set[str]) -> list[LayerFinding]:
    """One finding per layer, always all three.

    A layer the model omitted is silent, not missing. Dropping it would make the
    resolution look like it considered two layers when it considered three, and
    the trace is supposed to show what was rejected as well as what won.
    """
    by_layer = {}
    for item in raw:
        layer = item.get("layer")
        if layer in LAYERS and layer not in by_layer:
            by_layer[layer] = item

    findings = []
    for layer in LAYERS:
        item = by_layer.get(layer, {})
        outcome = item.get("outcome", "silent")
        if outcome not in ("grants", "denies", "silent"):
            outcome = "silent"
        rank = item.get("generosity_rank", NOT_APPLICABLE)
        citation = resolve_citation(item.get("citation", ""), valid_citations)

        # A layer claiming to speak without naming a provision has nothing
        # behind it. Treated as silent rather than allowed to control an answer
        # on the strength of an uncited assertion.
        if outcome != "silent" and citation is None:
            outcome = "silent"

        # A layer claiming to speak while carrying the not-applicable rank
        # sentinel cannot be compared against anything. Treated as silent
        # rather than coerced to rank 1, which is what `rank or 1` did:
        # it promoted malformed evidence to most-generous.
        if outcome != "silent" and (not isinstance(rank, int) or rank < 1):
            outcome = "silent"

        findings.append(
            LayerFinding(
                layer=layer,
                speaks_to_question=outcome != "silent",
                outcome=outcome,
                citation=citation,
                says=item.get("says", ""),
                generosity_rank=None if outcome == "silent" else rank,
            )
        )
    return findings


# At most two, so the answer stays readable. Ordered by retrieval rank, so the
# closest match to the question comes first.
MAX_HANDBOOK_NOTES = 2


def handbook_to_address(state: AgentState, resolution: Resolution) -> list[str]:
    """The handbook policies the answer has to acknowledge.

    **Not the same thing as "layers that lost".** `precedence.py` collects any
    speaking layer that did not control, which is usually a federal section, and
    a reader does not need told about a CFR provision they never opened. What
    they need is the handbook, because they have probably already read it and the
    answer is about to contradict it.

    Two cases the loser-based list could not reach:

    - The controlling layer beats the handbook and `precedence` names the losing
      *statute* instead, which is true and useless.
    - **The handbook is silent.** `conflict-005` asks about a sick grandparent:
      federal denies, Ohio adds nothing, the handbook says nothing at all. A
      silent layer never enters the comparison, so it can never be a loser, and
      yet "the handbook does not cover this" is precisely what the reader is
      trying to find out.

    So this reads the retrieved passages directly rather than the findings.
    """
    if "company" in resolution.defensible:
        return []

    seen: list[str] = []
    for hit in state.get("retrieved", []):
        if hit.authority_layer != "company":
            continue
        if hit.citation not in seen:
            seen.append(hit.citation)
        if len(seen) == MAX_HANDBOOK_NOTES:
            break
    return seen


def make_resolve(
    caller: StructuredCaller | None = None,
    model: str = HAIKU,
    passage_cap: int | None = DEFAULT_PASSAGE_CAP,
    temperature: float | None = DEFAULT_TEMPERATURE,
):
    caller = caller or StructuredCaller()

    def resolve(state: AgentState) -> dict:
        if not state.get("retrieved"):
            resolution = Resolution(
                controlling=None,
                rule="silence_is_not_permission",
                considered=[],
            )
            return {
                "resolution": resolution,
                "trace": [
                    TraceEvent(
                        node="resolve",
                        summary="nothing was retrieved, so no authority controls",
                        detail={"rule": resolution.rule},
                    )
                ],
            }

        # The cap is applied ONCE and everything downstream reads the capped
        # view. Capping only the prompt would leave `valid` containing citations
        # the model was never shown, so `resolve_citation` would accept a
        # citation that could not have been read, and `verify` would then check
        # the answer against evidence that never entered the decision.
        shown = cap_per_layer(state["retrieved"], passage_cap)
        seen_state: AgentState = {**state, "retrieved": shown}

        result = caller.call(
            system=f"{SYSTEM}\n\n<!-- {PROMPT_VERSION} -->",
            user=_user_message(seen_state),
            tool=READ_TOOL,
            model=model,
            max_tokens=2048,
            temperature=temperature,
        )
        valid = {h.citation for h in shown}
        findings = _to_findings(result.get("findings", []), valid)

        try:
            resolution = resolve_precedence(findings)
            # Handbook policies the reader has probably already opened come
            # first, ahead of any statute that merely lost the comparison.
            handbook = handbook_to_address(seen_state, resolution)
            others = [
                c for c in resolution.non_controlling_to_address if c not in handbook
            ]
            resolution = replace(
                resolution, non_controlling_to_address=handbook + others
            )
            summary = _summarise(resolution)
        except PrecedenceError as exc:
            # Inconsistent evidence is not resolved on a best guess. The answer
            # degrades rather than being asserted from findings that contradict
            # each other, which would still look like a resolution downstream.
            resolution = Resolution(
                controlling=None, rule="not_reached", considered=findings
            )
            summary = f"evidence was inconsistent, so no authority was resolved: {exc}"

        return {
            "resolution": resolution,
            "trace": [
                TraceEvent(
                    node="resolve",
                    summary=summary,
                    detail={
                        "controlling": resolution.controlling,
                        "acceptable": resolution.acceptable,
                        "rule": resolution.rule,
                        "considered": [
                            {
                                "layer": f.layer,
                                "outcome": f.outcome,
                                "citation": f.citation,
                                "says": f.says,
                                "generosity_rank": f.generosity_rank,
                            }
                            for f in resolution.considered
                        ],
                        "model": model,
                        # The trace has to show what the decision was allowed to
                        # see, not just what retrieval found, or a capped run and
                        # an uncapped one are indistinguishable after the fact.
                        "passage_cap": passage_cap,
                        "passages_shown": len(shown),
                        "passages_retrieved": len(state["retrieved"]),
                        "usage": getattr(caller, "last_call", None),
                    },
                )
            ],
        }

    return resolve


# Written for a non-technical reader, because the trace is a feature rather than
# a debug flag and this line is the one that explains the whole product.
_RULE_TEXT = {
    "statutory_floor": "federal and state both apply, and the more generous one governs",
    "policy_below_floor": (
        "the handbook promises less than the law requires, so the law governs"
    ),
    "policy_may_exceed": "company policy is more generous than the law requires, so it governs",
    "silence_is_not_permission": "only one layer addresses this, so it governs",
    "concurrence_tie_break": (
        "the handbook restates the statute, so the statute is what compels this"
    ),
    "indeterminate": "federal and state law each independently compel the same outcome",
    "not_reached": "no authority was resolved",
}


def _summarise(resolution: Resolution) -> str:
    reason = _RULE_TEXT.get(resolution.rule, resolution.rule)
    if resolution.controlling:
        return f"{resolution.controlling} law controls: {reason}"
    if resolution.acceptable:
        return f"{' and '.join(resolution.acceptable)} both control: {reason}"
    return reason
