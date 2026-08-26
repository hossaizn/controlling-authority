# Controlling Authority — project context

> Durable context for this repo. Read this before touching anything.
> The full reasoning behind every decision is in `eval/decision_log.md` (DL-1 to
> DL-20). This file is the summary; that file is the authority.

## What this is

An agent that answers employee leave questions by determining **which authority
controls** — federal, state, or company policy — then answering from it with
citations. Built as a portfolio piece for a Forward Deployed AI Engineer
application at Dayforce (HCM), so the domain is deliberately theirs.

**The premise, and it is measurable rather than rhetorical:** in this corpus the
correct answer frequently contradicts the *most semantically relevant document*.
The company handbook is the closest match and, where it falls below a statutory
floor, the wrong answer. No amount of retrieval tuning fixes that. It is a
reasoning problem over retrieved evidence.

Public repo: `github.com/hossaizn/controlling-authority`

## Status

Phases 0 through 5.5 complete and merged. Phase 6 (the agent) is next.

**Adopted configuration**, settled by measurement in DL-18 and revised in DL-20:

| choice | value | recall@10 |
|--------|-------|-----------|
| embedding | `voyage-law-2` (legal domain, 1024d) | |
| chunking | structure-aware | **0.895** |
| reranking | not built | |

Retrieval by slice: adversarial 1.000, control 1.000, superseded 1.000,
straightforward 0.941, **conflict 0.722**. The conflict gap is by design and is
what Phase 6 exists to close.

## Non-negotiables

**`corpus/handbook/DEFECTS.md` must never be ingested.** It states which handbook
clauses are deliberately wrong and how each resolves. Ingesting it puts the
answer key inside the corpus being searched, and every conflict scenario becomes
answerable by lookup. Two independent guards exist (filename allowlist with slug
denylist, plus a content check). Do not weaken either.

**Never pool verified and unverified scenarios into one number.** A citation
drafted from recall is not ground truth (DL-3). Averaging them reports a
measurement and a guess as one figure.

**Scripted edits must assert they matched.** `str.replace()` returns the string
unchanged when it finds nothing. DL-12: a decision-log entry described a
correction that never ran, and every artifact agreed with every other while all
were wrong.

**Tests pin values, not relationships.** DL-10: mutation testing broke the date
arithmetic three ways and the suite passed every time, because it asserted
`a >= b` on derived values.

## Architecture

```text
ingest/     four adapters, one SourceDocument contract
            federal_ecfr (XML API + amendment history)
            state_ca (server-rendered HTML), state_ny (authenticated JSON tree)
            company_handbook (Markdown), absence (recorded silences)
retrieval/  chunking (2 strategies), embed (providers), sparse, store (Qdrant), cache
eval/       scenarios (92, seven slices), metrics, run_retrieval, report, decision_log
corpus/     handbook policies, absence records, raw/ (gitignored cache)
```

**Corpus: 104 documents.** 79 federal (29 CFR 825), 6 California, 2 New York,
8 Ohio absence records, 9 handbook policies.

**Hybrid retrieval**: dense + sparse fused with RRF, filters applied to each
prefetch so they stay hard constraints. Jurisdiction and effective date are
**filters, never ranking signals** — a jurisdiction error is a correctness
failure, and a superseded provision is not a slightly worse answer.

**Absence records are documents, not config.** Ohio's silence is retrievable
text with `content_status: absent`, because a retrieval miss and a genuine
absence demand opposite responses and must not look alike.

## Method conventions that earned their place

**Pre-register predictions with a falsifier and a threshold.** DL-14 predicted
the chunking winner, its mechanism, and a tie-break, all before data existed.
The mechanism was then found wrong *before* the experiment (DL-15) and corrected,
which is the only reason the result means anything.

**Mutation-test anything numeric.** Every review that used it found real defects.

**Verify claims against ingested text, never recall.** DL-19: five ground-truth
errors, none of which would fail a plausibility check, all of which survived at
least one full review. Three were claims about *absent* law; one cited a real
section that did not say what was claimed.

**A negative claim cannot reach the same standard as a positive one.** You
cannot grep for the absence of a law. Absence records are marked "searched, not
found, scope stated" and dated, never ticked like positive claims.

**Search across titles.** All three false Ohio records were contradicted by
provisions outside the title being swept — one in the Administrative Code, one in
Title 23, one in Title 59.

## Gotchas that keep recurring

- **Bash cwd drifts between calls in this workspace.** Always `cd` explicitly in
  the same command.
- **Voyage is rate-limited to 3 req/min and 10K tokens/min without a payment
  method.** Pacing is proactive (`retrieval/ratelimit.py`); reactive backoff was
  not enough and killed one run 264 chunks in. Free tokens are unaffected: the
  whole experiment is under 1% of the allowance. **Deliberately no card on file
  — that is a hard spending cap of zero.**
- **Embeddings are cached on disk** by content hash and model. Re-runs are free;
  this turned a 66-minute run into 5 minutes.
- **Qdrant's in-memory mode silently ignores payload indexes.** Verify storage
  changes against the live server.
- **`.env` is edited in place, never appended.** Appending created a duplicate
  key and a misleading 401.

## Open items

- **Phase 6**: agent graph on Haiku. Upgrade a node to Sonnet only if its
  macro-averaged route accuracy is below **0.80**, fixed in advance.
- **`verify` is partly deterministic by design**, not a second LLM. Citation
  resolution, forbidden-citation checks and figure matching are code; only
  semantic entailment needs a model. This replaces DL-15's cross-family rule and
  is both cheaper and stronger.
- **Query rewriting is built in `triage` (DL-21), and it is narrower than DL-16
  claimed.** Date extraction is deliberately not built: the superseded pairs are
  word-identical questions differing only in `as_of_date`, so the date is an
  input, not something recoverable from text. Jurisdiction extraction is built
  but **no scenario exercises it**; 75 supply the state in context and none of
  the 17 that withhold it name a state in the question.
- **Retrieval numbers are raw-query numbers, not "oracle-filter" numbers.**
  DL-21 retracts that caveat: `employee_context` and `as_of_date` are ordinary
  caller inputs, and retrieval already runs unfiltered on the 10 scoreable
  scenarios that withhold a state. What still separates them from end-to-end is
  the rewritten query (watched by the regression gate) and routing errors
  (scored in `eval/run_routes.py`).
- **6 of 57 scoreable scenarios verified.** Ohio absences 4 of 8. Three
  verification boxes remain open, two for New York provisions no scenario needs.
- `codes.ohio.gov` has no programmatically reachable search, so a full-code
  keyword sweep is unavailable.
