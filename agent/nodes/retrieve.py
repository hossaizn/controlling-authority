"""`retrieve`: three-stage hybrid filtering.

    stage 1  pre-filter   indexed, selective, applied inside each prefetch
    stage 2  vector search dense + sparse, fused with RRF
    stage 3  post-filter  non-indexed conditions about the SET, not the chunk

Stages 1 and 2 live in `ChunkStore`. Jurisdiction and the as-of window are
payload-indexed and applied **inside each prefetch rather than after fusion**,
because a filter applied after RRF is a post-hoc trim that can silently return
fewer than `k` results, which reads as a worse answer rather than as a bug.

**Stage 3 is not here for latency.** Measured in DL-28: retrieval is 5.8 ms
against a 13.2 second end-to-end, four ten-thousandths of the budget, and
filtering costs nothing against not filtering. It is here for **guaranteed
presence**: some documents have to be in the candidate set for the agent to
reason correctly regardless of how they rank, and that is a condition on the set
rather than a property of any chunk.

Each guarantee over-fetches from a deeper search and appends. **Nothing is ever
displaced**, so the ranking the frozen regression baseline measures is untouched
and a guarantee can only add evidence, never remove it.

Raising `limit` to 30 would achieve the same reach and is the wrong trade: every
extra chunk is roughly 300 tokens in the `resolve` and `compose` prompts, which
are the real cost and the real latency. Twenty extra passages to surface one
document triples both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent.state import AgentState, TraceEvent
from retrieval.store import ChunkStore, SearchHit

DEFAULT_LIMIT = 10

# How deep to look when a guarantee is unmet. Costs one search, not one per
# guarantee: the query embedding is computed once.
LOOKAHEAD = 30


@dataclass(frozen=True)
class Guarantee:
    """A document class the candidate set must contain if one exists at all.

    `why` is written for the trace panel, because a reviewer seeing an extra
    passage appended needs to know it was deliberate.
    """

    name: str
    why: str
    holds_for: Callable[[SearchHit], bool]
    max_topup: int = 2


# The handbook guarantee was adopted in DL-27 on measured benefit: three metrics
# moved. It is the only one.
GUARANTEES: tuple[Guarantee, ...] = (
    Guarantee(
        name="handbook",
        why=(
            "the employee has probably already read the handbook, and part of "
            "the answer's job is to reconcile itself with what they read"
        ),
        holds_for=lambda h: h.authority_layer == "company",
    ),
)

# **Built, measured, and not adopted (DL-28).**
#
# Five of 33 Ohio answer scenarios have an absence record reachable at rank 30
# but not at rank 10, and rule 4 turns on telling a recorded silence apart from a
# retrieval miss. The mechanism was there and the argument was good.
#
# The pre-registered bar was an improvement of at least two scenarios. It
# delivered **zero**: precedence unchanged at 45 of 57 end to end and 50 of 57
# isolated, every other metric flat, no regression anywhere. So the agent was not
# losing those scenarios for want of the record.
#
# Kept here rather than deleted because it costs nothing to carry and the
# calculus changes with corpus size, but it is NOT in GUARANTEES and must not be
# added back without a number.
UNADOPTED_RECORDED_SILENCE = Guarantee(
    name="recorded_silence",
    why=(
        "rule 4 turns on telling a recorded silence apart from a retrieval miss, "
        "which cannot be done about a record that was never seen"
    ),
    holds_for=lambda h: h.content_status == "absent",
    max_topup=1,
)


def apply_guarantees(
    hits: list[SearchHit],
    deeper: list[SearchHit],
    guarantees: tuple[Guarantee, ...] = GUARANTEES,
) -> tuple[list[SearchHit], list[str]]:
    """Stage 3. Returns the augmented set and the names of guarantees that fired."""
    out = list(hits)
    seen = {h.chunk_id for h in out}
    fired: list[str] = []

    for guarantee in guarantees:
        if any(guarantee.holds_for(h) for h in out):
            continue
        extra = [
            h for h in deeper if h.chunk_id not in seen and guarantee.holds_for(h)
        ][: guarantee.max_topup]
        if not extra:
            # No such document exists within reach. Not an error: some questions
            # have no handbook policy and some jurisdictions have no absence
            # record on the topic.
            continue
        out.extend(extra)
        seen.update(h.chunk_id for h in extra)
        fired.append(guarantee.name)

    return out, fired


def make_retrieve(store: ChunkStore, limit: int = DEFAULT_LIMIT):
    def retrieve(state: AgentState) -> dict:
        query = state.get("rewritten_query") or state["question"]
        jurisdiction = state.get("jurisdiction")

        # Stages 1 and 2, searched deeper than returned so stage 3 has a pool.
        deeper = store.search(
            query,
            jurisdiction=jurisdiction,
            as_of=state["as_of"],
            limit=LOOKAHEAD,
        )
        ranked = deeper[:limit]

        # Stage 3.
        hits, fired = apply_guarantees(ranked, deeper)

        layers = sorted({h.authority_layer for h in hits})
        absent = [h.citation for h in hits if h.content_status == "absent"]

        return {
            "retrieved": hits,
            "trace": [
                TraceEvent(
                    node="retrieve",
                    summary=(
                        f"{len(hits)} passages across {len(layers)} authority "
                        f"layer(s), filtered to "
                        f"{jurisdiction or 'all jurisdictions'} "
                        f"as of {state['as_of'].isoformat()}"
                        + (f", topped up for: {', '.join(fired)}" if fired else "")
                    ),
                    detail={
                        "query": query,
                        "jurisdiction_filter": jurisdiction,
                        "as_of": state["as_of"].isoformat(),
                        "layers_found": layers,
                        "ranked": len(ranked),
                        "guarantees_fired": fired,
                        # A retrieval miss and a recorded silence demand opposite
                        # responses and must never look alike downstream.
                        "absence_records": absent,
                        "citations": [h.citation for h in hits],
                    },
                )
            ],
        }

    return retrieve
