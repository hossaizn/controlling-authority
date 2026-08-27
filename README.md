# Controlling Authority

An agent that answers employee leave questions by determining **which authority controls** — federal law, state law, or the company handbook — then answering from that authority with citations.

> *Controlling authority* is the legal term for the source that governs when several apply at once. Determining it is the actual work here. Retrieval is a means to that end.

**[→ Live demo](https://controlling-authority.pages.dev)**  ·  **[→ Decision log](https://controlling-authority.pages.dev/decisions.html)**

---

## The claim, and the measurement

On this corpus the correct answer frequently contradicts the *most semantically relevant document*. The handbook is the closest match to the question and, where it falls below a statutory floor, the wrong answer. No amount of retrieval tuning fixes that: it is a reasoning problem over retrieved evidence.

So precedence is implemented as **code**, not as a prompt, and measured against a system that trusts the top-ranked passage — same retrieval, same run, one component swapped:

| | trusts top passage | precedence as code |
|---|---|---|
| conflict slice (n=18) | 0.556 | **0.833** |
| control slice (n=10) | 0.300 | **0.800** |
| overall (n=57) | 0.649 | **0.877** |

**+22.8 points**, and the handbook is the top-ranked passage in 26 of 57 cases. The project's opening premise is a measured result rather than an assertion.

## End to end

`fully_correct` is a five-way conjunction: right route, right authority, required citations present, nothing forbidden cited, and grounded — all at once, across all 92 scenarios.

| metric | value |
|---|---|
| **fully correct** | **0.620** |
| route accuracy (macro) | 0.815 |
| precedence correct | 0.789 |
| verification pass rate | 0.672 *(n=58)* |
| forbidden citation rate | 0.011 |
| over-clarification | 0.052 |
| under-clarification | 0.467 |

Macro-averaged, never micro: a system that answers everything scores well on micro-average precisely by refusing to clarify. Over- and under-clarification are measured **in both directions**, because an agent that always asks *"which state?"* is trivially safe and unusable.

The weak numbers are here on purpose. `under_clarification` at 0.467 and `addressed_beaten_source` at 0.25 are the two worst results in the project, and they are reported beside the good ones.

## Read the decision log first

[`eval/decision_log.md`](eval/decision_log.md) — 39 entries, written as each decision was made rather than reconstructed afterwards. **Reverted experiments stay in.** A log containing only successes is a sales document.

A few that show the method:

- **[DL-14/15]** Pre-registered the chunking winner, its mechanism, and a tie-break before data existed. The mechanism was then found wrong *before* the experiment ran and corrected, which is the only reason the result means anything.
- **[DL-19/20]** Verification found five ground-truth errors, three of them false claims about *absent* law. The corrections flipped the chunking answer.
- **[DL-16/18]** Reranking was built, measured at 7.0 points of headroom against a pre-registered 10-point bar, and **not adopted**. Choosing a legal-domain embedding model had already removed the headroom a reranker would have chased.
- **[DL-28]** A retrieval guarantee was built, measured, and rejected: it delivered zero of a pre-registered two-scenario improvement. The code is kept, unwired, with a note that it must not be re-added without a number.
- **[DL-34]** My diagnosis of the verification bottleneck was wrong. Measurement overturned it, and the entry says so.
- **[DL-38]** A pre-registered experiment that has **not been run**, including the design that was rejected for being structurally biased toward its own conclusion.
- **[DL-39]** The production concerns this deliberately does not address, with the reasoning for each.

## How it decides

Precedence rules 1, 2, 4 and 5 are a pure function in [`agent/precedence.py`](agent/precedence.py). The model is asked only *what the retrieved text says* and *which provision is more generous*. It is never asked which layer controls.

1. **Statutory floor.** Where federal and state both apply, the more employee-favourable governs. Not "state beats federal."
2. **Company policy may exceed, never reduce.** Above statute it controls; below statute it is unenforceable.
3. **Effective dating.** Only provisions in force on the query date are eligible. *(Not reimplemented — the store's date filter already enforces it.)*
4. **Silence is not permission.** A layer that does not address a topic cannot override one that does.
5. **Concurrence tie-break.** A handbook restating a statute does not become the thing that compels it.

Rule 5 cannot be argued out of by a persuasively worded question, because it is not in the prompt.

## Architecture

```text
ingest/     four adapters, one SourceDocument contract
retrieval/  chunking, embeddings, sparse, Qdrant store, disk cache
agent/      triage → retrieve → resolve → compose → verify (LangGraph)
eval/       92 scenarios, seven slices, scorers, mutation harness, decision log
api/        FastAPI + a self-contained demo page
deploy/     static site build for Cloudflare Pages
```

- **Hybrid retrieval**: dense + sparse fused with RRF, filters applied *inside each prefetch* so they stay hard constraints rather than a post-hoc trim that silently returns fewer than `k`.
- **Jurisdiction and effective date are filters, never ranking signals.** A jurisdiction error is a correctness failure; a superseded provision is not a slightly worse answer.
- **Absence records are documents, not config.** Ohio's silence is retrievable text with `content_status: absent`, because a retrieval miss and a genuine absence demand opposite responses.
- **`verify` is mostly deterministic by design.** Four of its five checks are code; only semantic entailment needs a model.
- **The trace is append-only by construction**, via a LangGraph reducer, and is exported to Langfuse rather than instrumented a second time.

## Corpus

104 documents: 79 federal (29 CFR 825), 6 California, 2 New York, 8 Ohio absence records, 9 handbook policies.

| Layer | Source | Verified |
| --- | --- | --- |
| Federal | eCFR API, 29 CFR Part 825 (FMLA), point-in-time by snapshot date | 2026-08-25 |
| State | California (4 sections), New York (WCL 204), Ohio (recorded absences only) | 2026-08-26 |
| Company | Synthetic handbook, seeded with deliberate conflicts | n/a |

Ohio is the control case: no state family-leave statute for private employers. **A negative claim cannot reach the same standard as a positive one** — you cannot grep for the absence of a law — so absence records are marked *"searched, not found, scope stated"* and dated, never ticked like positive claims.

`corpus/handbook/DEFECTS.md` states which handbook clauses are deliberately wrong. **It is never ingested**, and two independent guards enforce that: ingesting the answer key would make every conflict scenario answerable by lookup.

## Testing

539 tests, and **mutation testing is the primary review technique** — every review that used it found real defects.

```bash
uv run pytest                    # 539 tests
uv run python -m eval.mutation   # 137 mutations, all currently caught
```

The mutation catalogue is committed because running it reactively kept finding holes that review did not: eval scorers that could all be hardcoded `True` with the suite green, an XSS guard that two different bypasses walked straight past, a path-traversal guard with no coverage, and a naive baseline that could be swapped for the real resolver without a single test failing — which is the control arm of the headline claim above.

**A stale mutation is an error, not a skip.** `str.replace()` returns the string unchanged when it finds nothing, so a mutation whose target has moved reports "caught" while never being applied.

## Running it

Requires Python 3.12, `uv`, and Docker.

```bash
cp .env.example .env      # fill in keys; never commit it
uv sync
docker compose up -d      # Qdrant on :6333
uv run pytest
uv run uvicorn api.app:app --reload    # demo at http://localhost:8000
```

The hosted demo is static, so its **Ask** box is disabled — new questions need a model call. Running locally enables it.

To rebuild the static site:

```bash
uv run python -m deploy.build_static   # writes dist/
```

## What this deliberately does not do

Recorded in full in DL-39. In short: no latency, throughput or availability SLOs were ever set — only quality ones, which were pre-registered and honoured. There is no checkpointer, no load test, no provider failover, and the rate limiter's in-memory counters are correct for one instance and wrong for two.

Each was raised, costed, and left undone on purpose. Naming them precisely is worth more here than closing them would be.

## Not legal advice

An information retrieval and reasoning system, not a compliance product. Scoped to leave and time-off only. The precedence rules are a simplification of real law.
