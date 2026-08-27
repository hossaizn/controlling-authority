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

from agent.citations import mentions, resolves_to_retrieved
from agent.models import HAIKU, StructuredCaller
from agent.nodes.compose import DISCLAIMER
from agent.state import AgentState, TraceEvent, VerificationResult
from ingest.settings import optional

PROMPT_VERSION = "verify-v1"


def entailment_blocks() -> bool:
    """Whether an unsupported claim discards the answer.

    Off by default. A referral saying "I could not confirm an answer" gives the
    reader nothing: no answer, no citations, no way to check. An answer with a
    flagged claim gives them the provision, the reasoning, and a specific thing
    to be careful about.

    `VERIFY_ENTAILMENT_BLOCKS=1` restores the strict posture, which a deployment
    weighing an unsupported claim as worse than no answer should set.
    """
    return bool(optional("VERIFY_ENTAILMENT_BLOCKS"))

# Same model as compose, for now. See the module docstring and DL-24.
VERIFY_MODEL = HAIKU

REFERRAL = (
    "I could not confirm an answer to this from the policies on record. "
    "Please ask your HR team."
)

# Only numbers carrying a UNIT are treated as quantities.
#
# The first design extracted every number and then tried to exclude the ones
# belonging to citations. A review showed that could not work: every federal
# citation is "29 CFR ...", so excluding digits found in citations made **29
# exempt corpus-wide**, and an answer claiming "you are entitled to 29 workweeks"
# passed the check. Meanwhile bare section references like "section 825.201" were
# flagged as fabricated quantities.
#
# Requiring a unit fixes both directions at once. A section number is never
# followed by "weeks"; a fabricated entitlement always is.
_QUANTITY = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:-\s*)?"
    r"(workweeks?|weeks?|calendar days?|business days?|days?|hours?|months?|"
    r"years?|employees?|persons?|percent|%)",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def _normalise(number: str) -> str:
    """`1,250` and `1250` are the same quantity. The corpus writes it one way and
    an answer the other, and a literal comparison called the correct figure
    fabricated."""
    return number.replace(",", "")

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
    return {c for c in known if mentions(text, c)}


def figures_in(text: str, known_citations: set[str]) -> set[str]:
    """Quantities the answer asserts: a number attached to a unit.

    Citations are stripped first so a policy id like `LEAVE-008` cannot supply a
    number, and then only number-plus-unit pairs are kept. See `_QUANTITY` for
    why the earlier "every number, minus citation digits" approach was unsound
    in both directions.
    """
    stripped = text
    for citation in known_citations:
        stripped = stripped.replace(citation, " ")
    return {_normalise(n) for n, _unit in _QUANTITY.findall(stripped)}


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

# Whole words only. Substring matching marked 10 as supported because "written"
# contains "ten", and 1 because "none" contains "one", which quietly turned a
# groundedness check into a rubber stamp on any corpus containing ordinary prose.
_WORD_NUMBER = re.compile(
    r"\b(" + "|".join(sorted(WORD_TO_DIGITS, key=len, reverse=True)).replace(" ", r"\s") + r")\b",
    re.IGNORECASE,
)


def supported_figures(corpus_text: str) -> set[str]:
    """Every quantity the sources state, in digits, however they were written."""
    found = {_normalise(n) for n in _NUMBER.findall(corpus_text)}
    for word in _WORD_NUMBER.findall(corpus_text):
        found.add(WORD_TO_DIGITS[" ".join(word.lower().split())])
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
        advisories: list[str] = []

        # 1. Nothing cited that was not retrieved.
        #
        # This checked `state["citations"]`, which `compose` has already filtered
        # by exactly the same predicate, so it could never fire. A review caught
        # it: deleting the failure branch changed no test. It now reads the
        # bracketed citations out of the ANSWER PROSE, which is what the original
        # comment claimed and the code did not do. That is the surface that
        # matters, because the prose is what the employee reads.
        in_prose = set(re.findall(r"\[([^\]\n]{2,60})\]", answer))
        stray = {
            c
            for c in in_prose | set(state.get("citations", []))
            if not resolves_to_retrieved(c, retrieved)
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
        # Where the resolution is indeterminate, ANY of the defensible layers'
        # provisions satisfies this. Taking only the first one failed an answer
        # that correctly cited state law because federal happened to come first
        # in `considered`, which is the same mistake as demanding a single
        # controlling authority when the spec says two are defensible.
        acceptable_citations = {
            f.citation
            for f in (resolution.considered if resolution else [])
            if f.layer in (resolution.defensible if resolution else []) and f.citation
        }
        checks["controlling_provision_cited"] = (
            bool(acceptable_citations & named) if acceptable_citations else True
        )
        if acceptable_citations and not (acceptable_citations & named):
            failures.append(
                "does not cite the controlling provision, expected one of "
                f"{sorted(acceptable_citations)}"
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
                notes = [
                    f"claim not supported by its source: {c}" for c in unsupported
                ] or ["a claim is not supported by its source"]
                # Advisory by default: this is a judgment call, and it was
                # discarding answers that passed every decidable check.
                (failures if entailment_blocks() else advisories).extend(notes)

        passed = not failures
        verification = VerificationResult(
            passed=passed, checks=checks, failures=failures, advisories=advisories
        )

        update: dict = {
            "verification": verification,
            "trace": [
                TraceEvent(
                    node="verify",
                    summary=(
                        f"failed: {failures[0]}"
                        if failures
                        else f"{sum(checks.values())}/{len(checks)} checks passed"
                        + (f", {len(advisories)} claim(s) flagged" if advisories else "")
                    ),
                    detail={
                        "checks": checks,
                        "failures": failures,
                        "advisories": advisories,
                        "entailment_checked": entailment_ran,
                        "model": model if entailment_ran else None,
                        "usage": getattr(caller, "last_call", None) if entailment_ran else None,
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
