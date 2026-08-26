"""`retrieve`: hybrid search under the filters triage established.

Thin on purpose. `ChunkStore` already applies jurisdiction and effective date as
hard filters inside each prefetch, so this node's whole job is to hand it the
right three inputs and record what came back.

**It sends the rewritten query, not the raw question.** That substitution is the
one thing standing between Phase 5's retrieval numbers and end-to-end, which is
why `eval/baseline_retrieval.json` was frozen before this node existed. DL-22
measured it: the rewrite moves recall@10 from 0.895 to 0.912 with no slice
regressing.

**A missing jurisdiction searches everywhere, and that is correct.** Substituting
a guess would filter out the law that actually governs, and the failure would be
silent because the result still looks like a result. Where the state genuinely
matters and is absent, the answer is a clarifying question, which triage has
already decided before this node runs.
"""

from __future__ import annotations

from agent.state import AgentState, TraceEvent
from retrieval.store import ChunkStore

DEFAULT_LIMIT = 10


def make_retrieve(store: ChunkStore, limit: int = DEFAULT_LIMIT):
    def retrieve(state: AgentState) -> dict:
        query = state.get("rewritten_query") or state["question"]
        jurisdiction = state.get("jurisdiction")

        hits = store.search(
            query,
            jurisdiction=jurisdiction,
            as_of=state["as_of"],
            limit=limit,
        )

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
                    ),
                    detail={
                        "query": query,
                        "jurisdiction_filter": jurisdiction,
                        "as_of": state["as_of"].isoformat(),
                        "layers_found": layers,
                        # Recorded because a retrieval miss and a recorded
                        # silence demand opposite responses and must never look
                        # alike downstream.
                        "absence_records": absent,
                        "citations": [h.citation for h in hits],
                    },
                )
            ],
        }

    return retrieve
