# Controlling Authority: design doc

*2026-08-25. Status: draft for review.*

---

## What this is

An agent that answers employee leave and benefits questions by determining **which authority controls**, then answering from that authority with citations.

"Controlling authority" is the legal term for the source that governs when several apply at once. It is the name of the product because determining it is the actual work. Retrieval is a means to that end.

A question like *"how much parental leave do I get?"* has no answer without knowing the employee's state, their tenure, their employer's size, and whether the company handbook promises more than statute requires. A system that retrieves a relevant passage and summarises it will be confidently wrong most of the time.

## Why this shape

Three properties make this worth building rather than another document-Q&A demo:

**The right answer often contradicts the most relevant document.** When a company handbook promises less than state law requires, the handbook is the best semantic match and the wrong answer. Naive RAG fails this by construction, not by tuning.

**Sometimes the correct output is a question, not an answer.** If jurisdiction is unknown and the answer differs by jurisdiction, asking is correct. If the answer is the same everywhere, asking is a failure. Both directions are measurable.

**Sometimes the correct output is a refusal.** "No policy covers this, here is who to ask" is a real answer. Most systems hallucinate instead.

## Non-goals

Cut deliberately, to keep the project finishable:

- **Not legal advice.** Every response carries a disclaimer. This is an information retrieval and reasoning system, not a compliance product.
- **No authentication or user accounts.** Employee context is supplied per request.
- **No conversation memory beyond a single session.**
- **No fine-tuning.** Off-the-shelf models throughout.
- **No coverage beyond leave and time-off.** Not payroll, not benefits enrolment, not tax.
- **Three states only.** California, New York, Ohio.

---

## Corpus

Three layers with a precedence relationship. Every chunk carries the metadata needed to reason about which layer it belongs to and when it applied.

### Layer 1: Federal

29 CFR Part 825 (FMLA) and related parts, from the [eCFR API](https://www.ecfr.gov/developers/documentation/api/v1).

Section-level XML with real hierarchy, so structure-aware chunking has structure to exploit. The API supports point-in-time retrieval on a snapshot date, which means **effective dating is demonstrated against real regulatory history rather than synthetic timestamps.**

### Layer 2: State

Chosen for contrast, not convenience:

| State | Why |
|---|---|
| California | Far exceeds federal. CFRA, PFL, paid sick leave. |
| New York | Paid family leave via a different mechanism, insurance-based. |
| Ohio | Essentially the federal minimum. The control case. |

Three jurisdictions is enough to make metadata filtering load-bearing and few enough that ingestion does not consume the project.

Each state publishes differently, so each gets its own ingestion adapter. **This is deliberately the messy part.** Heterogeneous source ingestion is the honest daily reality of forward-deployed work, and hiding it would make the project less representative, not more.

### Layer 3: Company handbook

Authored by us as Markdown with front matter. Realistic in tone: vague where real handbooks are vague, silent on things real handbooks omit.

**Seeded with deliberate defects**, each one a test case:

1. A promise that falls **below** California statutory minimum. Statute controls; the handbook is unenforceable on that point.
2. A promise that **exceeds** statute. The handbook controls, because policy may exceed the floor.
3. A clause **superseded** by a newer version, with both retained and effective-dated.
4. A topic the handbook simply **does not cover**, to force refusal or escalation.
5. A clause that is **ambiguous** about which employees it applies to, to force clarification.

### Chunk metadata schema

The metadata is what makes the vector store do real work rather than act as a lookup table.

```jsonc
{
  "chunk_id":        str,
  "authority_layer": "federal" | "state" | "company",
  "jurisdiction":    "US" | "CA" | "NY" | "OH",
  "source_id":       str,          # e.g. "29 CFR 825.200"
  "section_path":    list[str],    # hierarchy, for structure-aware chunking
  "effective_from":  date,
  "effective_to":    date | null,  # null means currently in force
  "citation":        str,          # human-readable, rendered in answers
  "source_url":      str,
  "text":            str
}
```

---

## Retrieval

Solid, not the research focus. Decisions to be validated by eval rather than asserted here.

**Hybrid dense plus sparse.** Legal text is dense with exact tokens (`FMLA`, `825.200`, `Section 125`) where embeddings underperform and keyword matching excels. Qdrant supports both natively in one query.

**Hard metadata filters before scoring.** A question about Ohio must never surface California law. This is a filter, never a ranking signal, because a jurisdiction error is a correctness failure, not a relevance failure.

**Effective-date filtering** at query time, defaulting to today and overridable to demonstrate point-in-time answers.

**Structure-aware chunking**, splitting on the regulatory hierarchy rather than fixed token windows, with the section path preserved. To be compared against a naive fixed-size baseline in the eval, with the winner recorded in the decision log.

---

## The agent

A LangGraph state machine. Explicit nodes, because a reviewer can read a graph and cannot read a prompt loop.

```text
                  ┌─────────────┐
                  │   triage    │
                  └──────┬──────┘
             ┌───────────┼───────────┐
             ▼           ▼           ▼
       ┌─────────┐ ┌──────────┐ ┌─────────┐
       │ clarify │ │ retrieve │ │ refuse  │
       └─────────┘ └────┬─────┘ └─────────┘
                        ▼
                 ┌─────────────┐
                 │  resolve    │   ← precedence reasoning
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  compose    │   ← answer + citations
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   verify    │   ← groundedness gate
                 └─────────────┘
```

**triage.** Classifies the request. Is it in scope? Is enough context present to answer? Does the answer vary by a fact we do not have?

**clarify.** Emits a single targeted question. Fires only when the missing fact would change the answer. Asking when it would not is a scored failure.

**retrieve.** Hybrid search under jurisdiction and date filters, across all three layers.

**resolve.** The core. Determines controlling authority per the rules below, and records why.

**compose.** Drafts the answer with citations to the controlling source, and notes when a non-controlling source says something different, because an employee who read the handbook deserves to know why the answer differs from it.

**verify.** Checks every claim is supported by a retrieved chunk. On failure, degrades to refusal rather than shipping an ungrounded answer.

**refuse.** States that no covering policy was found and routes to a human.

### Precedence rules

Applied by `resolve`, in order:

1. **Statutory floor.** Where federal and state both apply, the more generous to the employee governs. This is the rule people get wrong: it is not "state beats federal," it is "the employee-favourable floor wins."
2. **Company policy may exceed, never reduce.** A handbook term more generous than statute controls. A term less generous is unenforceable and statute controls.
3. **Effective dating.** Only provisions in force on the query date are eligible. Superseded text is retained for point-in-time queries and excluded otherwise.
4. **Silence is not permission.** A layer that does not address the topic does not override one that does.

Every resolution emits a structured trace: which layers were considered, which won, which rule decided it.

---

## Evaluation

The spine of the project. If the agent is the product, agent decisions are what get measured.

### Scenario schema

```jsonc
{
  "scenario_id":     str,
  "question":        str,
  "employee_context": { "state": str|null, "tenure_months": int|null,
                        "hours_worked_12mo": int|null, "employer_size": int|null },
  "as_of_date":      date,
  "expected_route":  "answer" | "clarify" | "refuse" | "escalate",
  "expected_authority": "federal" | "state" | "company" | null,
  "required_citations": list[str],
  "forbidden_citations": list[str],   # e.g. superseded or wrong-jurisdiction
  "notes":           str
}
```

Target roughly 80 to 120 scenarios, hand-written, spread across slices: straightforward, ambiguous jurisdiction, handbook conflict, superseded policy, out of scope, and adversarial.

### Metrics

| Metric | What it catches |
|---|---|
| Route accuracy | Wrong high-level behaviour |
| Precedence correctness | Right answer from the wrong authority, which is luck |
| Citation groundedness | Claims not supported by retrieved text |
| Forbidden-citation rate | Superseded or wrong-jurisdiction sources leaking in |
| Refusal correctness | Hallucinating instead of declining |
| **Over-clarification rate** | Asking when the answer would not change |
| Retrieval recall@k | Whether the ceiling is retrieval or reasoning |

**Over-clarification is the metric that shows judgment.** An agent that always asks is trivially safe and unusable. Measuring both failure directions is the point.

**Groundedness is scored by a separate model** with the claim and the retrieved chunks, never by the model that wrote the answer. Self-grading is not evaluation.

### The decision log

One entry per experiment, fixed shape: hypothesis, metric watched, result, decision, effect. Committed as results arrive, not written retrospectively.

**Failed experiments stay in.** A reverted change with the number that killed it is the strongest evidence the log is real.

---

## Observability

Langfuse tracing on every stage. Each request records node path, retrieval filters and hits, precedence resolution, token cost and latency per stage, and final route.

The trace is surfaced in the demo UI, not hidden behind a debug flag. Explaining an AI system to non-AI stakeholders is the job this project is auditioning for.

---

## Demo surface

Public, live, no auth.

- **Six curated scenarios as buttons.** Straightforward, ambiguous jurisdiction, handbook conflict, superseded policy, refusal, escalation. A reviewer sees something non-obvious without having to invent a good question.
- **Baseline toggle.** The same question answered by naive RAG and by the full agent, side by side. On the handbook-conflict case the baseline confidently returns the wrong answer. **This single feature is the argument.**
- **Live trace panel.** Graph execution, retrieval, precedence, cost.
- **Free-text input**, so it can be probed and broken.
- **As-of date picker**, to demonstrate point-in-time answers.

### Protection layer

Public inference is exposed spend. Mirrors the architecture already running in production on the Primrose & Eve Workers:

- Per-IP rate limiting
- Per-session daily quota
- **Global daily circuit breaker**, sized so the worst case bounds spend rather than draining the budget
- Hard input length cap
- **Pre-computed responses for the six curated scenarios**, so the common path costs nothing and returns instantly
- No secrets in client code

---

## Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python | Closes a genuine portfolio gap |
| API | FastAPI | Known, fast, typed |
| Vector store | Qdrant | Real payload filtering, native hybrid |
| Agent | LangGraph | Explicit, readable state machine |
| Tracing | Langfuse | Open source, self-hostable, good traces |
| Eval | Custom harness, Ragas where it fits | Scenario schema is bespoke |
| Packaging | Docker Compose | One command to run locally |

---

## Repo layout

```text
controlling-authority/
├── README.md                 decision log first, setup last
├── ingest/                   per-source adapters
│   ├── federal_ecfr.py
│   ├── state_ca.py  state_ny.py  state_oh.py
│   └── company_handbook.py
├── corpus/                   handbook source, cached raw pulls
├── retrieval/                chunking, embedding, hybrid search
├── agent/                    graph, nodes, precedence rules
├── eval/
│   ├── scenarios/            the golden set
│   ├── metrics/
│   └── decision_log.md
├── api/                      FastAPI service, protection layer
├── ui/                       demo, trace panel, baseline toggle
└── docker-compose.yml
```

---

## Risks

**Ingestion sprawl.** State law is heterogeneous and could eat the schedule. Mitigation: fixed cap of three states, and a hard rule that ingestion stops when the eval set is answerable.

**Scenario authoring is slow.** 80 to 120 hand-written scenarios with correct citations is real work and cannot be generated without destroying the ground truth. Mitigation: write scenarios first, in slices, and let corpus needs follow from them.

**Precedence rules oversimplify real law.** They do. Mitigation: the disclaimer, and scoping to leave only.

**Public demo cost.** Mitigated by the protection layer above.

---

## Resolved decisions

**Embedding: hosted.** Local models add container weight and cold-start latency that would ruin first impressions on a cheap tier. **Which hosted model is decision log entry #1**, not an assertion: embed the corpus with a general-purpose model and with a legal-domain model, measure retrieval recall on the eval set, keep the winner, record the result. The project demonstrates its own methodology on its own first choice.

**Generation: multi-model routing.** Cheap model for `triage`, strong reasoning model for `resolve` and `compose`. Mirrors the pattern already running in production on the Primrose & Eve Workers.

**`verify` uses a different model family than `compose`.** A model checking its own output shares its blind spots. Cross-family verification is a materially stronger groundedness gate than self-checking.

**Hosting: managed free tiers plus one container.** Three self-hosted services is neither free nor necessary.

| Service | Where | Why |
|---|---|---|
| Qdrant | Qdrant Cloud free tier | Removes the heaviest container |
| Langfuse | Langfuse Cloud free tier | Removes the second |
| FastAPI app | Fly.io, single small instance | The only thing that must be ours |

Docker Compose stays for **local development**, so a reviewer can run the full stack with one command. Production is a single service.

Inference still costs money. The protection layer plus pre-computed curated scenarios bounds it, and the path most reviewers take never calls a model.

**Date picker: ship the capability, not the control.** A free date picker requires the reviewer to know which date is interesting, which violates the ninety-second rule. Instead one curated scenario runs the same question at two dates side by side and shows the answer change across a supersession. A free picker is a second-pass feature.

The principle, applied throughout: **invisible sophistication is wasted sophistication.** Same reason the trace panel is a feature rather than a debug flag.
