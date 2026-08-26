# Controlling Authority

An agent that answers employee leave questions by determining **which authority controls**, then answering from that authority with citations.

> *Controlling authority* is the legal term for the source that governs when several apply at once. Determining it is the actual work here. Retrieval is a means to that end.

**Status: in development.** Phase 0 of 10. This README grows as results arrive, and the decision log below is written as experiments run, not retrospectively.

---

## The problem

"How much parental leave do I get?" has no answer on its own. It depends on the employee's state, their tenure, their employer's size, and whether the company handbook promises more than statute requires.

Three properties make this harder than document Q&A:

**The right answer often contradicts the most relevant document.** When a handbook promises less than state law requires, the handbook is the best semantic match and the wrong answer. Naive RAG fails this by construction, not by tuning.

**Sometimes the correct output is a question.** If the answer varies by jurisdiction and jurisdiction is unknown, asking is correct. If the answer is identical everywhere, asking is a failure. Both directions are measured here.

**Sometimes the correct output is a refusal.** "No policy covers this, here is who to ask" is a real answer. Most systems hallucinate instead.

## How it decides

Precedence is applied in order:

1. **Statutory floor.** Where federal and state both apply, the more employee-favourable governs. It is not "state beats federal."
2. **Company policy may exceed, never reduce.** A handbook term above statute controls. Below statute, it is unenforceable and statute controls.
3. **Effective dating.** Only provisions in force on the query date are eligible.
4. **Silence is not permission.** A layer that does not address a topic does not override one that does.

## Corpus

| Layer | Source | Verified |
|-------|--------|----------|
| Federal | eCFR API, 29 CFR Part 825 (FMLA). Point-in-time by snapshot date. | 2026-08-25 |
| State | California (4 sections), New York (WCL 204), Ohio (recorded absences only, no statute cited by any scenario) | 2026-08-26 |
| Company | Synthetic handbook, seeded with deliberate conflicts | n/a |

Ohio is the control case: it has no state family-leave statute for private employers. That absence is encoded as an explicit record, because a retrieval miss and a genuine absence must not look the same to the agent.

---

## Decision log

Written as each decision is made, not reconstructed afterwards. **Reverted experiments stay in.** Full entries with reasoning are in [`eval/decision_log.md`](eval/decision_log.md).

| | Decision | Status |
|---|---|---|
| **DL-1** | Legal-domain vs general-purpose embeddings, measured before either is adopted | open, Phase 5 |
| **DL-2** | Ground truth authored before the pipeline exists, so the eval cannot be circular | decided |
| **DL-3** | Statutory ground truth verified against ingested text, never asserted from recall | decided, verification pending |
| **DL-4** | Scenario slice balance committed before any result is visible | decided |
| **DL-5** | Every conflict case paired with a control case, so caution is not free | decided |
| **DL-6** | Keep New York for its integration shape, not its coverage; adapter scoped to the 7 scenarios | decided |
| **DL-7** | Independent review found the ground truth partly unscoreable; three missing rules, not bad data | decided |
| **DL-8** | Effective dating from amendment history, not the snapshot date | decided |
| **DL-9** | Federal citations verified against ingested text; one claim was too broad | federal done |
| **DL-10** | Mutation testing showed the date tests proved nothing; contract was federal-shaped | decided |
| **DL-11** | Verification found two ground-truth errors; premature verification now enforced, not remembered | Phase 3 |
| **DL-12** | A log entry described work that never ran; silent no-ops now assert | decided |

---

## Evaluation

Ninety-two hand-written scenarios, each carrying a question, employee context, an as-of date, and the labelled correct behaviour. Written **before** the retrieval pipeline, because they cannot be generated without destroying their own ground truth, and because they are what tells ingestion when to stop.

Measured: route accuracy, precedence correctness, citation groundedness, forbidden-citation rate, refusal correctness, **over-clarification rate**, retrieval recall@k.

Over-clarification is the metric that matters most for judgment. An agent that always asks "which state?" is trivially safe and unusable.

Groundedness is scored by a different model than the one that wrote the answer. Self-grading is not evaluation.

---

## Running it

Requires Python 3.12 and Docker.

```bash
cp .env.example .env      # fill in keys
uv sync                   # install
docker compose up -d      # Qdrant on :6333
uv run pytest             # tests
```

New York's Open Legislation API needs a free key from <https://legislation.nysenate.gov/>. Federal, California and Ohio sources need no credentials.

---

## Not legal advice

An information retrieval and reasoning system, not a compliance product. Scoped to leave and time-off only. Precedence rules here are a simplification of real law.
