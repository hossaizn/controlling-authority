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

**Phases 0 through 7 merged.** Phase 8 (API + protection layer) is next, then
9 (demo) and 10 (ship).

**Tracing is exported, not instrumented (Phase 7).** Nodes stay unaware of
Langfuse; `agent/tracing.py` walks the finished state trace and mirrors it.
A second instrumentation path would be a second description of one run.
`export` never raises, and **evals deliberately do not trace** because they
run almost entirely from cache and would bury real traces in spans for
decisions that never called a model. `run_traced` is the request boundary.

**End to end, DL-25 and DL-26:** fully correct **0.620** across all 92
scenarios, meaning right route, right authority, required citations present,
nothing forbidden and grounded, all at once. Verification 0.672 over the 58 that
reach it, precedence 0.789. Weakest slice is conflict at 0.278.

**The bottleneck is verification strictness (DL-27).** A third of answers
are replaced by a referral, taking their citations with them, and most of
those failures are entailment self-grading on the model that wrote the
answer. DL-24's open-weights arm is the pre-registered test for it and is
now the highest-value work left.

**Every eval scorer is now tested.** A Phase 6 review found they could all be
mutated to return True with the suite still green (DL-26). Treat
`eval/run_*.py` as production code: it is what every reported number rests on.

**DL-24 is resolved, partially.** `openai/gpt-oss-120b` matches Haiku on routing
(**0.818 vs 0.815** macro, at **$0.00**) and beats it on `clarify` (0.800 vs
0.533). Precedence is **unmeasured and not measurable on a free tier**: 23% of
`resolve` prompts exceed Groq's 8,000-token per-request ceiling, and the excluded
ones are systematically the evidence-heaviest, so scoring the remainder would
flatter the open model.

**Free-tier constraints, none of them in the rate-limit headers.** 200k tokens
per model per day; reserved `max_tokens` charged against the limit rather than
tokens produced; reasoning tokens billed against `max_tokens` and emitted before
the tool call; `gpt-oss-20b` cannot honour `tool_choice`. Do not plan a run
without accounting for all four.

**Still open:** the entailment check self-grades on the same model as `compose`,
which accounts for 20 of 26 verification failures. DL-24's open arm was meant to
supply a cross-family verifier and can, since `verify` prompts are small enough
to fit the ceiling.

**The headline, DL-23.** Precedence as code against a system that trusts the
top-ranked passage, same retrieval, same run:

| | naive | precedence as code |
|---|---|---|
| conflict slice | 0.500 | **0.833** |
| overall | 0.632 | **0.877** |

The handbook is the top-ranked passage 26 times of 57. The spec's opening claim
is now a measured +24.6 points rather than a premise.

**Triage, DL-22:** route accuracy **0.815 macro**, clearing the 0.80 upgrade
threshold by less than one clarify scenario. Weakest route is clarify at 0.533.
Query rewriting **improves** retrieval, 0.895 to **0.912** recall@10.

**Precision at n=57 is about one scenario.** Three resolve prompts scored 0.860,
0.860, 0.877 and are indistinguishable. Do not quote the spread as progress.

**Adopted configuration**, settled by measurement in DL-18 and revised in DL-20:

| choice | value | recall@10 |
|--------|-------|-----------|
| embedding | `voyage-law-2` (legal domain, 1024d) | |
| chunking | structure-aware | **0.895** |
| reranking | not built | |

Retrieval by slice: adversarial 1.000, control 1.000, superseded 1.000,
straightforward 0.941, **conflict 0.722**. The conflict gap is by design and is
what Phase 6 exists to close.

## Blocked, as of 2026-08-26

**The Anthropic account is out of API credits.** Calls return
`invalid_request_error: credit balance is too low`. Note the Console's "Usage
credits" panel (plan overage) is a different balance from prepaid API credits; a
promotional credit may sit in one and not the other.

Consequences: cached decisions still serve, so the retrieval gate and any run
whose prompts are unchanged still work. Anything that changed a prompt or a
candidate set needs credits to reproduce.

**DL-24's open-weights arm is now the unblocking move, not just the interesting
one.** `agent/models.py` speaks Anthropic and any OpenAI-compatible endpoint;
set `OPEN_MODEL_API_KEY`, `OPEN_MODEL_ID` and `OPEN_MODEL_BASE_URL`, then:

```bash
uv run python -m eval.run_triage      "$OPEN_MODEL_ID"
uv run python -m eval.run_precedence  "$OPEN_MODEL_ID"
uv run python -m eval.run_end_to_end  "$OPEN_MODEL_ID"
```

The provider must honour `tool_choice`. Forced structured output is the contract
every node depends on, and a provider that answers in prose fails loudly rather
than being parsed.

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
