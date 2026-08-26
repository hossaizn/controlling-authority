"""The precedence rules, as code.

**This module is the project.** Everything else is scaffolding around it.

The premise the whole thing rests on is that in this corpus the correct answer
frequently contradicts the most semantically relevant document: the handbook is
the closest match and, where it falls below a statutory floor, the wrong answer.
Phase 5 measured that directly. On the conflict slice the handbook ranks *first*
on precisely the scenarios where it is wrong, and retrieval sits twenty points
below every other slice as a result. No embedding model fixes it, because the
handbook genuinely is the closest match.

**So precedence is a function, not a prompt.** Given a set of findings about what
each authority layer says, which one controls is deterministic. A model is asked
exactly two things, both of them about reading the evidence rather than applying
the rules:

1. does this provision speak to the question at all, and what does it say
2. which provision is more generous to the employee, where that is not arithmetic

It is never asked which layer controls. That is decided here, where the rule that
fired is recorded, the same input always produces the same output, and the
subtlest rule in the spec, that a handbook restating a statute *concurs* rather
than controls, cannot be talked out of by a persuasive question.

**Rule 3, effective dating, is not implemented here.** It has already run: the
store applies the as-of date as a hard filter inside each prefetch, so superseded
text never reaches these findings. Re-checking it here would be a second
implementation of one rule, and the two would eventually disagree.
"""

from __future__ import annotations

from agent.state import LayerFinding, PrecedenceRule, Resolution
from domain import Authority

# Federal and state are statutes. Rule 5 orders statute above policy, and
# deliberately does NOT order federal against state: where both independently
# compel the same outcome, neither is "the" controlling authority.
STATUTORY: tuple[Authority, ...] = ("federal", "state")


class PrecedenceError(ValueError):
    """Raised when the findings are internally inconsistent.

    Loud rather than silent. A resolution computed from contradictory evidence
    still looks like a resolution, and nothing downstream could tell.
    """


def _validate(findings: list[LayerFinding]) -> None:
    seen = [f.layer for f in findings]
    if len(seen) != len(set(seen)):
        raise PrecedenceError(f"one finding per layer; got {seen}")

    for f in findings:
        if f.outcome != "silent" and f.generosity_rank is None:
            raise PrecedenceError(
                f"{f.layer} {f.outcome} but carries no generosity_rank; it cannot "
                "be compared against the other layers"
            )

    # A layer that denies cannot be more generous than one that grants. The
    # ranking comes from a model, and this is the one way it can be checked
    # against itself: if it were wrong here, a handbook that grants leave would
    # lose to a statute that does not require it, and the employee would be told
    # no on the strength of a rule that was only ever a floor.
    grants = [f.generosity_rank for f in findings if f.outcome == "grants"]
    denies = [f.generosity_rank for f in findings if f.outcome == "denies"]
    if grants and denies and min(denies) <= min(grants):
        raise PrecedenceError(
            "a layer that denies is ranked at least as generous as one that "
            f"grants: grants={grants}, denies={denies}"
        )


def resolve_precedence(
    findings: list[LayerFinding],
    non_controlling_to_address: list[str] | None = None,
) -> Resolution:
    """Apply spec rules 1, 2, 4 and 5 to a set of layer findings."""
    _validate(findings)
    considered = list(findings)
    to_address = list(non_controlling_to_address or [])

    # Rule 4: silence is not permission. A layer that does not address the topic
    # is removed from contention entirely rather than treated as permitting.
    # Filters on `outcome` alone. `speaks_to_question` mirrors it and is kept
    # for the trace, but two fields encoding one fact can disagree, and a
    # review found nothing exercised the difference.
    speaking = [f for f in findings if f.outcome != "silent"]

    if not speaking:
        return Resolution(
            controlling=None,
            rule="silence_is_not_permission",
            considered=considered,
            non_controlling_to_address=to_address,
        )

    best = min(f.generosity_rank for f in speaking)  # type: ignore[type-var]
    winners = [f for f in speaking if f.generosity_rank == best]

    statutes_speaking = [f for f in speaking if f.layer in STATUTORY]
    company_speaking = [f for f in speaking if f.layer == "company"]

    winning_statutes = [f for f in winners if f.layer in STATUTORY]
    company_won = any(f.layer == "company" for f in winners)

    rule, survivors = _decide(
        winners=winners,
        winning_statutes=winning_statutes,
        company_won=company_won,
        statutes_speaking=statutes_speaking,
        company_speaking=company_speaking,
    )

    # Anything that spoke and did not control is worth naming to the reader, and
    # the handbook especially: they have probably already read it.
    losers = [
        f.citation
        for f in speaking
        if f not in survivors and f.citation and f.citation not in to_address
    ]

    if len(survivors) > 1:
        return Resolution(
            controlling=None,
            rule="indeterminate",
            considered=considered,
            acceptable=[f.layer for f in survivors],
            non_controlling_to_address=to_address + losers,
        )

    return Resolution(
        controlling=survivors[0].layer,
        rule=rule,
        considered=considered,
        non_controlling_to_address=to_address + losers,
    )


def _decide(
    winners: list[LayerFinding],
    winning_statutes: list[LayerFinding],
    company_won: bool,
    statutes_speaking: list[LayerFinding],
    company_speaking: list[LayerFinding],
) -> tuple[PrecedenceRule, list[LayerFinding]]:
    """Which rule fired, and who survives it."""

    # Rule 5, the subtle one, and the one a review found missing entirely (DL-7).
    # A handbook term that merely matches a statutory floor is CONCURRING, not
    # controlling: strike the handbook and the entitlement survives untouched.
    # Before this rule existed the same restatement pattern was labelled
    # `federal` in one scenario and `company` in another, and precedence was
    # unscoreable.
    if winning_statutes and company_won:
        return "concurrence_tie_break", winning_statutes

    if company_won:
        # Rule 2: policy may exceed the floor. It won outright, so either it beat
        # a statute or no statute addresses the topic at all.
        if statutes_speaking:
            return "policy_may_exceed", winners
        return "silence_is_not_permission", winners

    # A statutory layer won.
    if company_speaking:
        # Rule 2's other half: a handbook term below the floor is unenforceable
        # and the statute controls. This is the case the demo's baseline toggle
        # exists to show, because naive retrieval returns the handbook first.
        #
        # Labelled separately from rule 1. Both were reported as
        # `statutory_floor`, which merged the count of the project's central
        # case with an unrelated federal-versus-state comparison, so the one
        # counter that evidences the thesis was not measuring it.
        return "policy_below_floor", winning_statutes

    if len(statutes_speaking) > 1:
        # Rule 1: federal and state both apply and the employee-favourable one
        # governs. Not "state beats federal".
        return "statutory_floor", winning_statutes

    # Only one layer spoke at all, so nothing was overridden.
    return "silence_is_not_permission", winning_statutes
