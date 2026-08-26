"""`verify`: check the answer is grounded, mostly without a model.

**Replaces the cross-family rule the spec originally set (DL-15).** The reasoning
that made that rule attractive, that a model checking its own output shares its
blind spots, is right; the conclusion, that the fix is a second model from
another vendor, was weaker than it looked. Code cannot share a blind spot with a
model, because it is not reasoning. Four of the five checks here are functions.

- every citation in the answer was actually retrieved
- the controlling provision is among them
- every figure quoted appears in the retrieved text
- an answer asserting an entitlement cites something

Only entailment, whether each claim follows from the passage it cites, needs a
model, and it is the last line rather than the whole gate.

**On failure the answer degrades to a referral.** Shipping an ungrounded answer
is worse than shipping no answer, because the reader cannot tell the difference.

**A known weakness, recorded rather than hidden.** The entailment call currently
runs on the same model that wrote the answer, which is the self-grading the spec
warns about. It is deliberate for now: DL-24 pre-registers an open-weights arm,
and that model is a cross-family verifier for free once it exists.
"""

from __future__ import annotations

import re

from agent.models import HAIKU, StructuredCaller
from agent.nodes.compose import DISCLAIMER
from agent.state import AgentState, TraceEvent, VerificationResult

PROMPT_VERSION = "verify-v1"

# Same model as compose, for now. See the module docstring and DL-24.
VERIFY_MODEL = HAIKU

REFERRAL = (
    "I could not confirm an answer to this from the policies on record. "
    "Please ask your HR team."
)

# Figures that carry no factual weight: ordinals and small counts appear in
# ordinary prose ("the first of these", "both parents") and flagging them would
# make the check noisy enough to be ignored, which is worse than not having it.
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")

ENTAILMENT_TOOL = {
    "name": "check_grounding",
    "input_schema": {
        "type": "object",
        "properties": {
            "every_claim_supported": {"type": "boolean"},
            "unsupported": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Each claim in the answer that the passages do not support.",
            },
        },
        "required": ["every_claim_supported", "unsupported"],
    },
}

SYSTEM = """You check whether an answer is supported by the passages it cites.

You are not judging whether the answer is good, well written, or helpful. You are
judging one thing: does every factual claim in it follow from the supplied text?

A claim is UNSUPPORTED when the passages do not say it, even if it is true in
general or sounds reasonable. Plausible is not supported.

A claim is SUPPORTED when the passages say it, including when they say it in
different words.

Ignore the closing disclaimer. Ignore statements about who to contact. Ignore
the answer telling the reader what a source says and why it does not apply,
provided the passages support what that source says."""


def citations_in(text: str, known: set[str]) -> set[str]:
    """Which known citations the answer actually names.

    Matched against the retrieved set rather than parsed out of the prose,
    because a regex for "things that look like a citation" would both miss
    handbook ids and invent matches from ordinary numbers.
    """
    return {c for c in known if c in text}


def figures_in(text: str, known_citations: set[str]) -> set[str]:
    """Numbers the answer asserts as quantities, with citations removed first.

    Two false positives had to be closed, both found by this check failing valid
    answers rather than by reasoning about it.

    **Section numbers are not quantities.** Stripping the full citation string is
    not enough: an answer that cites `[29 CFR 825.201]` and then refers to
    "section 825.201" leaves a bare `825.201` behind, which was then reported as
    a figure the sources do not support. Any number that appears inside a known
    citation is excluded.

    **Statutes spell numbers out.** The corpus says "twenty-six weeks" where the
    answer says "26 weeks", so a literal digit search reported a correct figure
    as unsupported. Word forms are converted before the comparison, in
    `supported_figures`.
    """
    stripped = text
    for citation in known_citations:
        stripped = stripped.replace(citation, " ")

    citation_digits = {
        n for c in known_citations for n in _NUMBER.findall(c)
    }
    return {n for n in _NUMBER.findall(stripped) if n not in citation_digits}


_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# "twenty-six" -> "26", "twelve" -> "12". Legal drafting spells numbers out far
# more often than it uses digits, and the corpus is legal drafting.
WORD_TO_DIGITS: dict[str, str] = {w: str(i) for i, w in enumerate(_UNITS)}
for _word, _value in _TENS.items():
    WORD_TO_DIGITS[_word] = str(_value)
    for _i, _unit in enumerate(_UNITS[1:10], start=1):
        WORD_TO_DIGITS[f"{_word}-{_unit}"] = str(_value + _i)
        WORD_TO_DIGITS[f"{_word} {_unit}"] = str(_value + _i)


def supported_figures(corpus_text: str) -> set[str]:
    """Every quantity the sources state, in digits, however they were written."""
    lowered = corpus_text.lower()
    found = set(_NUMBER.findall(corpus_text))
    for word, digits in WORD_TO_DIGITS.items():
        if word in lowered:
            found.add(digits)
    return found


def make_verify(caller: StructuredCaller | None = None, model: str = VERIFY_MODEL):
    caller = caller or StructuredCaller()

    def verify(state: AgentState) -> dict:
        answer = state.get("answer") or ""
        hits = state.get("retrieved", [])
        resolution = state.get("resolution")
        retrieved = {h.citation for h in hits}
        corpus_text = "\n".join(h.text for h in hits)

        named = citations_in(answer, retrieved)
        checks: dict[str, bool] = {}
        failures: list[str] = []

        # 1. Nothing cited that was not retrieved. `compose` already strips these
        # from the citation list, so this catches them appearing in the prose.
        stray = {
            c for c in state.get("citations", []) if c not in retrieved
        }
        checks["citations_were_retrieved"] = not stray
        if stray:
            failures.append(f"cites provisions that were not retrieved: {sorted(stray)}")

        # 2. An answer that asserts an entitlement has to rest on something.
        asserts_entitlement = bool(resolution and resolution.defensible)
        checks["answer_is_cited"] = bool(named) or not asserts_entitlement
        if asserts_entitlement and not named:
            failures.append("asserts an entitlement without citing any provision")

        # 3. The provision the precedence rules selected is the one the answer
        # should rest on. An answer that reaches the right outcome by citing a
        # different layer is right by luck.
        controlling_citation = None
        if resolution:
            controlling_citation = next(
                (
                    f.citation
                    for f in resolution.considered
                    if f.layer in resolution.defensible and f.citation
                ),
                None,
            )
        checks["controlling_provision_cited"] = (
            controlling_citation in named if controlling_citation else True
        )
        if controlling_citation and controlling_citation not in named:
            failures.append(
                f"does not cite the controlling provision {controlling_citation!r}"
            )

        # 4. Every quoted figure appears in the text it came from.
        stated = supported_figures(corpus_text)
        unsupported_figures = sorted(
            f for f in figures_in(answer, retrieved) if f not in stated
        )
        checks["figures_appear_in_the_sources"] = not unsupported_figures
        if unsupported_figures:
            failures.append(f"quotes figures not in the sources: {unsupported_figures}")

        # 5. The only check that needs a model, and the last one rather than the
        # gate. Skipped when the deterministic checks have already failed: there
        # is no point paying to grade an answer that is being discarded.
        entailment_ran = False
        if not failures and named:
            entailment_ran = True
            cited_text = "\n\n".join(
                f"[{h.citation}] {h.text}" for h in hits if h.citation in named
            )
            result = caller.call(
                system=f"{SYSTEM}\n\n<!-- {PROMPT_VERSION} -->",
                user=f"Answer:\n{answer}\n\nPassages it cites:\n\n{cited_text}",
                tool=ENTAILMENT_TOOL,
                model=model,
            )
            supported = bool(result.get("every_claim_supported"))
            checks["claims_follow_from_the_sources"] = supported
            if not supported:
                unsupported = [c for c in result.get("unsupported", []) if c][:3]
                failures.extend(
                    f"claim not supported by its source: {c}" for c in unsupported
                )
                if not unsupported:
                    failures.append("a claim is not supported by its source")

        passed = not failures
        verification = VerificationResult(
            passed=passed, checks=checks, failures=failures
        )

        update: dict = {
            "verification": verification,
            "trace": [
                TraceEvent(
                    node="verify",
                    summary=(
                        f"{sum(checks.values())}/{len(checks)} checks passed"
                        if passed
                        else f"failed: {failures[0]}"
                    ),
                    detail={
                        "checks": checks,
                        "failures": failures,
                        "entailment_checked": entailment_ran,
                        "model": model if entailment_ran else None,
                    },
                )
            ],
        }

        if not passed:
            # Degrade rather than ship. The reader cannot tell a grounded answer
            # from an ungrounded one, which is exactly why this cannot be left
            # to their judgment.
            update["answer"] = f"{REFERRAL}\n\n{DISCLAIMER}"
            update["citations"] = []

        return update

    return verify
