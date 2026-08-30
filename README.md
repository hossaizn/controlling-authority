# Controlling Authority

An agent that answers employee leave questions by working out **which authority controls**, federal law, state law, or the company handbook, then answering from that authority with citations.

**[Live demo](https://controlling-authority.pages.dev)** · **[Decision log](https://controlling-authority.pages.dev/decisions)** · **[Implementation plan](https://controlling-authority.pages.dev/plan)**

---

## What this is, and why this problem

Ask how much leave you get and three authorities might answer. Federal law sets a floor. Your state may set a higher one. Your company handbook is the document that actually mentions your situation by name.

The handbook is almost always the closest semantic match to the question. When it promises less than the law requires, the closest match is the wrong answer, and nothing in the text of any of the three says which one wins.

I picked this because better retrieval does not fix it. Fetching the right passage is the easy half. Something still has to decide which authority governs, and I wanted to find out whether that decision belongs in code rather than in a prompt.

It does, and the difference is measurable. Same retrieval, same runs, one component swapped:

| | trusts the closest passage | precedence in code |
|---|---|---|
| conflict slice (n=18) | 0.556 | **0.833** |
| control slice (n=10) | 0.300 | **0.800** |
| overall (n=57) | 0.649 | **0.877** |

**+22.8 points.** The handbook is the top-ranked passage in 26 of those 57 cases.

End to end across all 92 scenarios, `fully_correct` is 0.620. That figure is a five-way conjunction: right route, right authority, required citations present, nothing forbidden cited, and grounded, all at once.

---

## What it is built from

Every technique below is one this project actually uses. The ones it does not use are listed further down.

### Retrieval

**Hybrid search with reciprocal rank fusion.** Dense vectors and sparse term vectors run as separate prefetches and fuse with RRF. Statutory text is full of terms of art that carry exact meaning, and a dense embedding places `825.200` near its paraphrases, which is the opposite of what a citation lookup needs. Sparse matching finds the literal token. See [`retrieval/store.py`](retrieval/store.py) and [`retrieval/sparse.py`](retrieval/sparse.py).

**Metadata filters as hard constraints, applied inside each prefetch.** Jurisdiction and effective date are filters, never ranking signals. A jurisdiction error is a correctness failure, and a superseded provision is not a slightly worse answer. Applying the filter after fusion would silently return fewer than `k` results, which reads as a worse answer rather than as a bug. Payload indexes back both fields.

**Domain-specific embeddings, chosen by measurement.** A legal-domain model and a general-purpose one were both embedded and scored before either was adopted. The legal model won at 0.895 recall@10. See DL-1 and DL-18.

**Structure-aware chunking, also chosen by measurement**, against a fixed-size baseline. Headings and hierarchy travel with the chunk, so a bare subdivision is not stranded from the section that gives it meaning. [`retrieval/chunking.py`](retrieval/chunking.py) carries both strategies, because the comparison stays reproducible.

**Query rewriting**, folded into the routing step rather than run as a separate call. Measured at 0.895 to 0.912 recall@10, and watched by a regression gate. See [`agent/nodes/triage.py`](agent/nodes/triage.py) and DL-21.

**A third filtering stage for guaranteed presence.** Some documents have to be in the candidate set regardless of rank, which is a condition on the set rather than a property of any chunk. Each guarantee over-fetches from a deeper search and appends, so nothing is displaced. [`agent/nodes/retrieve.py`](agent/nodes/retrieve.py) also carries one guarantee that was built, measured, and left unwired because it delivered zero of a pre-registered improvement.

**Absence is a document, not a config flag.** Ohio has no state family-leave statute for private employers, and that silence is retrievable text carrying `content_status: absent`. A retrieval miss and a genuine absence demand opposite responses and must not look alike to the agent. See [`ingest/absence.py`](ingest/absence.py).

### The agent

**Routing before retrieval.** A LangGraph state machine sends each question to answer, clarify, refuse, or escalate. Sometimes the correct output is a question. Sometimes it is a refusal. Both are scored. See [`agent/graph.py`](agent/graph.py).

**The precedence rules are code, not a prompt.** This is the whole thesis. [`agent/precedence.py`](agent/precedence.py) is a pure function. The model is asked only what the retrieved text says and which provision is more generous to the employee. It is never asked which layer controls. The rule that fired is recorded, the same input always gives the same output, and a persuasively worded question cannot argue past a rule that is not in the prompt.

**Forced tool calls for structured output.** Every node that needs a shape gets one through a required tool call rather than by parsing prose. A provider that cannot honour `tool_choice` fails loudly instead of being parsed. See [`agent/models.py`](agent/models.py).

**Groundedness verification that is mostly deterministic.** Four of the five checks in [`agent/nodes/verify.py`](agent/nodes/verify.py) are code: citations resolve to something retrieved, an entitlement claim rests on a citation, the answer cites the provision precedence selected, and every quoted figure appears in its source. Only semantic entailment needs a model. Code cannot share a blind spot with the model that wrote the answer.

**Citation validation against the retrieved set.** Model output is mapped back to a citation that was actually retrieved, longest match first, and rejected if it was not. See [`agent/nodes/resolve.py`](agent/nodes/resolve.py) and [`agent/citations.py`](agent/citations.py).

**Provider-agnostic model access.** [`agent/models.py`](agent/models.py) speaks Anthropic and any OpenAI-compatible endpoint, which is what made it possible to measure an open-weights model against a hosted one without touching a node.

### Evaluation

**92 hand-written scenarios across seven slices**, written before the retrieval pipeline existed. Ground truth cannot be generated by the system it is meant to evaluate without becoming circular. See [`eval/scenarios/`](eval/scenarios/).

| slice | n | what it tests |
| --- | --- | --- |
| conflict | 18 | the closest match is the wrong answer |
| straightforward | 17 | one authority answers cleanly |
| ambiguous | 15 | the correct output is a question |
| out_of_scope | 12 | refusal and escalation |
| adversarial | 10 | prompts that argue for the wrong layer |
| control | 10 | paired with conflict, so caution is not free |
| superseded | 10 | point-in-time answering |

**An ablation with a real control arm.** The naive baseline is the same graph with precedence swapped out. One component, same retrieval, same run. Two separate implementations would differ for reasons nobody intended. See [`agent/build.py`](agent/build.py).

**Pre-registration with falsifiers and thresholds fixed before data exists.** DL-14 predicted the chunking winner, its mechanism and a tie-break in advance. The mechanism was then found wrong before the experiment ran and corrected, which is the only reason the result means anything.

**Macro-averaged routing accuracy, never micro.** A system that answers everything scores well on a micro average precisely by refusing to clarify. Over-clarification and under-clarification are measured in both directions, because an agent that always asks "which state?" is trivially safe and unusable. See [`eval/run_routes.py`](eval/run_routes.py).

**Per-slice regression gates.** An overall number that holds steady hides a slice that collapsed. See [`eval/regression.py`](eval/regression.py).

**Mutation testing as the primary review technique.** [`eval/mutation.py`](eval/mutation.py) breaks the source on purpose, 180 ways, and checks whether the suite notices. Every review that ran it found real defects, including eval scorers that could all be hardcoded to `True` with the whole suite green. A stale mutation counts as an error, not a skip: `str.replace()` returns the string unchanged when it finds nothing, so a mutation whose target moved reports "caught" while never having been applied.

```bash
uv run pytest                    # 634 tests
uv run python -m eval.mutation   # 180 mutations, all currently caught
```

### Running it in production terms

**Tracing is exported, not instrumented.** [`agent/tracing.py`](agent/tracing.py) walks the finished state trace and mirrors it to Langfuse. Nodes stay unaware of it. A second instrumentation path would be a second description of one run, free to disagree with the first.

**Prompt versioning.** Each node carries a version string that keys the cache and labels the run, so a scored result cannot be quietly attributed to a prompt that has since changed.

**Content-hash caching of model calls.** Re-runs are free, which is why a killed evaluation resumes without paying twice.

**Spend limits that are enforced rather than hoped for.** [`api/limits.py`](api/limits.py) has sliding-window rate limits per IP and per session, plus a global daily breaker. Budget is charged in a `finally`, because an earlier version recorded only on success, and a graph that raised after calling the model spent real tokens against an untouched counter.

---

## What this does not use

Named because the list matters as much as the one above.

- **No corrective RAG or self-RAG.** The graph does not grade its own retrieval and re-query. Verification runs after composition and degrades the answer to a referral rather than looping.
- **No reranker, and that decision is currently reopened.** DL-16 fixed a rule in advance: build one only if `recall@10` minus `recall@3` exceeded 10 points. On raw questions the gap is 7.0, so none was built, and the reason was interesting rather than marginal: choosing a legal-domain embedding model removed the headroom a reranker would have chased. Then query rewriting was adopted, and on rewritten questions the gap is 14.0, because rewriting raises `recall@10` and lowers `recall@3`. The rule fires on the pipeline as shipped. DL-40 records that, and it was found while writing this README rather than by a review.
- **No multi-hop or iterative retrieval.** One retrieval pass per question.
- **No fine-tuning, no HyDE, no GraphRAG, no knowledge graph.**
- **No agentic tool loop.** The graph is fixed and every path through it is enumerable, which is what makes the trace worth reading.
- **No semantic caching.** The cache is exact content hashing.

---

## Choices, including the ones I skipped

A reviewer will look for these. Each is either a thing this project does, with the reasoning, or a thing it does not, with the reason.

### Latency

Measured once, reactively, when a filtering decision needed it. Qdrant search runs at **5.8 ms** median against a **13,198 ms** end-to-end, so retrieval is **0.04%** of the budget. Perfecting it would save 6 ms out of 13 seconds.

The cost is four sequential model calls, and no amount of retrieval tuning moves that. Two things were done about it. The graph is built once at startup rather than per request, and the curated scenarios are pre-computed, which takes the path most reviewers walk down to zero model calls and zero wait.

**No latency target was ever set.** Had a p95 been fixed on day one, the serial chain would have been a design constraint rather than an observation. DL-28 and DL-39.

### Token cost

Retrieval returns 10 passages, not the 30 it searches. Every extra chunk is roughly 300 tokens in the `resolve` and `compose` prompts, so fetching 30 to surface one document triples the bill and the latency.

`verify` sends targeted evidence rather than everything: about 1,000 tokens instead of 6,700, selected from the citations actually under test.

DL-38 asked the sharper question, what is the minimum context `resolve` needs, and pre-registered an experiment with thresholds. It is built and **not yet run**, because it needs either credits or a free-tier window that has not reset.

### Guardrails

Layered, and most of them are code rather than prompt instructions.

- **Forced tool calls.** Every structured output comes back through a required tool. A provider that answers in prose fails loudly instead of being parsed.
- **Citation validation.** Model output is mapped to a citation that was actually retrieved, and rejected otherwise.
- **Precedence in code.** A persuasively worded question cannot argue past a rule that is not in the prompt. The adversarial slice tests exactly that.
- **Spend limits.** Sliding-window caps per IP and per session, a global daily breaker, and an input length cap. Budget is charged in a `finally`, because an earlier version charged only on success and a mid-graph failure spent real tokens against an untouched counter.
- **Corpus integrity.** `DEFECTS.md` is the answer key and two independent guards keep it out of the ingested corpus.
- **The public page.** One sink where data reaches the DOM, no HTML built from strings, and a build-time check for off-origin loads.

### Hallucination

`verify` is the guard, and four of its five checks need no model: citations resolve to retrieved text, an entitlement claim rests on a citation, the answer cites the provision precedence chose, and every quoted figure appears in its source. Only entailment uses a model.

The figures check took three attempts. It exempted `29` corpus-wide because every federal citation starts with it, matched `"ten"` inside `"written"`, and compared `1,250` against `1250` as different numbers. DL-35.

A failed check degrades the answer to a referral rather than shipping it. That costs the reader the citations, which is the honest downside, and it is reported at 0.672 rather than hidden.

### Sampling parameters

**Temperature is settable and ships unset.** That is now a measured choice rather than the gap DL-41 first recorded it as. No top-p, top-k, frequency or presence penalty.

For `triage` the standard choice is temperature 0, and DL-41 pre-registered a prediction that setting it would move routing by less than 2 points. Both arms then ran on `openai/gpt-oss-120b` at $0.00, paired scenario by scenario:

| | provider default | temperature 0 |
|---|---|---|
| macro accuracy | 0.8178 | **0.8440** |
| micro accuracy | 0.8913 | 0.8913 |
| scenarios correct | 82 / 92 | 82 / 92 |

**The prediction is refuted, and the improvement is not real.** Macro moved 2.62 points while not one additional question was answered correctly. Four scenarios were fixed and four were broken, and macro averages the four routes equally, so one scenario is worth `(1/n)/4` macro points and routes with few scenarios dominate:

| route | n | one scenario, in macro points | change | contribution |
|---|---|---|---|---|
| escalate | 6 | 4.17 | +1 | **+4.17** |
| refuse | 14 | 1.79 | +1 | **+1.79** |
| clarify | 15 | 1.67 | -2 | **-3.33** |
| answer | 57 | 0.44 | 0 | 0.00 |
| | | | | **+2.62** |

The whole delta is four scenarios landing in three routes of different sizes. A scenario in `escalate` is worth **9.5 times** one in `answer`.

That is a caution about reading any macro delta here, including the 0.815 that clears this project's own 0.80 threshold.

Temperature 0 also raised under-clarification from 0.200 to 0.333 while over-clarification did not move: the model answered ambiguous leave questions instead of asking one. So it was **not adopted**, and `DEFAULT_TEMPERATURE` stays `None`.

Two things this does not claim. Haiku is unmeasured, so the case for temperature 0 on the shipped model stands untested. `resolve` and `verify` are unmeasured, because the precedence arm needs a provider whose per-request ceiling clears 8,000 tokens. 23% of `resolve` prompts exceed Groq's, and the ones that fail are systematically the evidence-heaviest, so scoring the rest flatters the model rather than measuring it (DL-24, DL-42).

The disk cache is why runs reproduce, and **that reproducibility is caching, not determinism**. Sampling is now part of the cache key, encoded so that an unset call hashes exactly as it did before, which is why adding it invalidated none of the 1,127 cached decisions.

### Data drift

**Provenance drift is handled. Statistical drift is not.**

Pre-computed records carry the prompt and corpus versions they were generated under, and a record generated under different ones is served marked stale rather than passed off as current. Prompt versions key the cache. Corpus documents carry snapshot dates, and absence records carry a `verified_on` date and the scope that was searched.

There is no monitoring of embedding or query distribution over time, because the corpus is a fixed 104-document snapshot and nothing arrives after ingestion. On a live corpus that would be the first thing to add.

### Deployment

The demo ships as static files on Cloudflare Pages: the six scenarios, both comparison arms and the traces are all pre-computed, and the argument the project makes needs no server.

Free container hosting that stays awake without a credit card ended during 2026. Fly retired its free tier, Hugging Face moved Docker Spaces to a paid plan, and the rest want a card. A server whose one dynamic feature is rate-limited into single digits a day was not worth a cold start on every first click. DL-39.

### Knowledge graph

Not used, and not close. Three authority layers with a precedence order between them is a rule set, not a graph problem. The relationships that matter are "is more generous than" and "was in force on", and both are computed from fields the documents already carry.

---

## Corpus

104 documents: 79 federal (29 CFR 825), 6 California, 2 New York, 8 Ohio absence records, 9 handbook policies.

| layer | source | verified |
| --- | --- | --- |
| federal | eCFR API, 29 CFR Part 825, point-in-time by snapshot date | 2026-08-25 |
| state | California, New York WCL 204, Ohio recorded absences | 2026-08-26 |
| company | synthetic handbook, seeded with deliberate conflicts | n/a |

**A negative claim cannot reach the same standard as a positive one.** You cannot grep for the absence of a law, so absence records are marked "searched, not found, scope stated" and dated, never ticked like a positive claim.

`corpus/handbook/DEFECTS.md` records which handbook clauses are deliberately wrong. It is never ingested, and two independent guards enforce that. Putting the answer key inside the corpus being searched would make every conflict scenario answerable by lookup.

---

## Repository map

```text
ingest/      four source adapters behind one SourceDocument contract
retrieval/   chunking, embeddings, sparse vectors, Qdrant store, disk cache
agent/       triage, retrieve, resolve, compose, verify, plus precedence.py
eval/        92 scenarios, scorers, regression gate, mutation harness, decision log
api/         FastAPI service and the self-contained demo page
deploy/      static site build for Cloudflare Pages
docs/        the implementation plan and the design spec
```

Start with [`agent/precedence.py`](agent/precedence.py) and [`eval/decision_log.md`](eval/decision_log.md).

---

## Running it

Requires Python 3.12, `uv`, and Docker.

```bash
cp .env.example .env      # fill in keys; never commit it
uv sync
docker compose up -d      # Qdrant on :6333
uv run pytest
uv run uvicorn api.app:app --reload    # demo at http://localhost:8000
```

The hosted demo is static, so its Ask box is off. A new question needs a model call. Running locally switches it on.

To rebuild the static site:

```bash
uv run python -m deploy.build_static   # writes dist/
```

---

## What this deliberately does not address

DL-39 has it in full. No latency, throughput or availability targets were ever set, only quality ones, which were pre-registered and honoured. There is no checkpointer, no load test, no provider failover, and the rate limiter's in-memory counters are correct for one instance and wrong for two.

Each was raised, costed, and left undone on purpose.

---

## Not legal advice

An information retrieval and reasoning system, not a compliance product. Scoped to leave and time off. The precedence rules are a simplification of real law.
