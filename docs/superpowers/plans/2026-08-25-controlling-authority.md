# Controlling Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A public, traced agent that answers HR leave questions by determining which authority controls, with an eval harness and a decision log proving the design choices.

**Architecture:** Three-layer corpus (federal / state / company) in Qdrant with jurisdiction and effective-date metadata. LangGraph agent routes between clarify, retrieve, resolve, compose, verify and refuse. FastAPI service behind a rate-limited public demo.

**Tech Stack:** Python 3.12, FastAPI, Qdrant, LangGraph, Langfuse, pytest, Docker Compose (local), Fly.io (production).

**Spec:** `docs/superpowers/specs/2026-08-25-controlling-authority-design.md`

---

## Source verification (done 2026-08-25)

Checked live before writing this plan. These are facts, not assumptions.

| Source | Status | Notes |
|--------|--------|-------|
| eCFR | **Verified** | Official API. `GET /api/versioner/v1/full/{date}/title-29.xml?part=825` returned 350KB of structured XML. Title 29 current to 2026-08-21. Point-in-time by date works. |
| California | **Verified** | `leginfo.legislature.ca.gov` is server-rendered. Full text of Gov Code 12945.2 present in the HTTP response. Needs an HTML parser, no JS. |
| Ohio | **Verified** | `codes.ohio.gov` is server-rendered. Statute body present in raw HTML. **Pages carry `Effective: <date>`**, which feeds `effective_from` directly. |
| New York | **Verified** | Open Legislation API, key obtained. `GET /api/3/laws/WKC?depth=2` returns a navigable tree. Paid Family Leave sits in **Article A9, Disability Benefits**. Nodes carry `activeDate`, which feeds `effective_from`. `docType`/`locationId` map to `section_path`. |

**Ohio has no state family-leave statute for private employers.** That is why it is the control case. The absence must be encoded explicitly (Task 3.3) or it cannot be tested.

---

# PHASE 0: Foundations

## Task 0.1: Repo and skeleton

**Files:** Create `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `docker-compose.yml`

- [ ] **Step 1: Init the repo**

```bash
cd job-search/dayforce-ai-engineer/controlling-authority
git init && git branch -M main
```

- [ ] **Step 2: `.env.example` with placeholders only**

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
VOYAGE_API_KEY=
NY_SENATE_API_KEY=
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

Never commit `.env`. `.gitignore` must contain `.env`, `__pycache__/`, `.venv/`, `corpus/raw/`.

- [ ] **Step 3: Verify no secrets can be committed**

```bash
git check-ignore -v .env && echo "ignored, good"
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore .env.example README.md docker-compose.yml
git commit -m "chore: project skeleton"
```

## Task 0.2: Register for the New York API key

- [ ] Sign up at `https://legislation.nysenate.gov/`, put the key in `.env`.
- [ ] Verify:

```bash
python -c "import os,urllib.request;k=os.environ['NY_SENATE_API_KEY'];print(urllib.request.urlopen(f'https://legislation.nysenate.gov/api/3/laws/WKC?key={k}').status)"
```

Expected: `200`. If this blocks, drop NY and run with CA and OH. Two contrasting states still make jurisdiction filtering load-bearing.

---

# PHASE 1: Handbook and scenarios FIRST

**This phase comes before any retrieval code.** The scenario set is the largest piece of human effort, cannot be generated without destroying its own ground truth, and tells ingestion when to stop. Building the pipeline first is how projects like this end up with twelve smoke tests instead of an eval.

## Task 1.1: Author the company handbook

**Files:** Create `corpus/handbook/*.md`

- [ ] **Step 1: Write 8 to 12 policy sections** as Markdown with front matter:

```markdown
---
policy_id: LEAVE-004
title: Parental Leave
effective_from: 2025-01-01
effective_to: null
supersedes: null
---
```

- [ ] **Step 2: Seed the five deliberate defects.** Each is a test case:

1. Below California minimum (statute must override)
2. Above statute (handbook must control)
3. Superseded, both versions retained with dates
4. Topic absent entirely (forces refusal)
5. Ambiguous applicability (forces clarification)

Record which policy carries which defect in `corpus/handbook/DEFECTS.md`. **That file is ground truth and must never be fed to the agent.**

- [ ] **Step 3: Commit**

## Task 1.2: Scenario schema and the first slice

**Files:** Create `eval/scenarios/schema.py`, `eval/scenarios/straightforward.yaml`, `tests/test_scenario_schema.py`

- [ ] **Step 1: Write the failing test**

```python
def test_scenario_requires_expected_route():
    with pytest.raises(ValidationError):
        Scenario(scenario_id="s1", question="q", as_of_date=date(2026,1,1))
```

- [ ] **Step 2: Run it, confirm it fails** (`pytest tests/test_scenario_schema.py -v`)

- [ ] **Step 3: Implement the Pydantic model** per the spec's scenario schema. `expected_route` is a required Literal of `answer|clarify|refuse|escalate`.

- [ ] **Step 4: Run, confirm pass**

- [ ] **Step 5: Write 15 straightforward scenarios**, answerable from one authority with no conflict.

- [ ] **Step 6: Commit**

## Task 1.3: The remaining scenario slices

**Files:** `eval/scenarios/{ambiguous,conflict,superseded,out_of_scope,adversarial}.yaml`

- [ ] Ambiguous jurisdiction, 15. Answer differs by state, state not supplied. Expect `clarify`.
- [ ] **Control cases, 10.** Answer identical across states. Expect `answer`, never `clarify`. These are what make over-clarification measurable.
- [ ] Handbook conflict, 20. Split between handbook-below-statute and handbook-above-statute.
- [ ] Superseded, 10. Same question at two `as_of_date` values with different correct answers.
- [ ] Out of scope, 10. Payroll, tax, benefits enrolment. Expect `refuse`.
- [ ] Adversarial, 10. Prompt injection, requests for legal advice, hypotheticals. Expect `refuse` or `escalate`.
- [ ] **Step 7: Assert the set is balanced**

```python
def test_scenario_slices_are_balanced():
    counts = Counter(s.expected_route for s in load_all())
    assert counts["clarify"] > 0 and counts["refuse"] > 0
    assert counts["answer"] >= counts["clarify"]  # or the agent learns to always ask
```

- [ ] **Step 8: Commit**

---

# PHASE 2: Federal ingestion

## Task 2.1: eCFR adapter

**Files:** Create `ingest/federal_ecfr.py`, `tests/test_federal_ecfr.py`

- [ ] **Step 1: Write the failing test** against a checked-in XML fixture, not the network:

```python
def test_parses_section_hierarchy():
    sections = parse_ecfr_xml(FIXTURE)
    s = next(x for x in sections if x.doc_id == "us:29-cfr-825.200")
    assert s.section_path == ["Part 825", "Subpart B"]  # ancestors only
    assert s.authority_layer == "federal"
    assert s.jurisdiction == "US"
    assert s.citation == "29 CFR 825.200"
```

- [ ] **Step 2: Save the fixture**

```bash
curl -s "https://www.ecfr.gov/api/versioner/v1/full/2026-08-01/title-29.xml?part=825" \
  > tests/fixtures/ecfr_825.xml
```

- [ ] **Step 3: Implement the parser.** Walk `DIV5/DIV6/DIV8`, emit one record per section with `section_path` preserved. Set `effective_from` from the snapshot date.

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

## Task 2.2: Point-in-time pull

- [ ] **Step 1: Test that two snapshot dates can differ**

```python
def test_snapshot_date_is_recorded():
    recs = fetch_part(825, as_of=date(2020,1,1))
    assert all(r.effective_from <= date(2020,1,1) for r in recs)
```

- [ ] **Step 2: Implement with on-disk caching** in `corpus/raw/` (gitignored). Never hit the API twice for the same date.
- [ ] **Step 3: Commit**

---

# PHASE 3: State ingestion

Each state gets its own adapter. Heterogeneous ingestion is the honest reality of forward-deployed work.

## Task 3.1: California

**Files:** Create `ingest/state_ca.py`, `tests/test_state_ca.py`

- [ ] Fixture-based test asserting Gov Code 12945.2 body text extracts cleanly and `jurisdiction == "CA"`.
- [ ] Parse server-rendered HTML. Verified present, no JS needed.
- [ ] Target sections: CFRA (12945.2), pregnancy disability leave, paid sick leave.
- [ ] Commit.

## Task 3.2: New York

**Files:** Create `ingest/state_ny.py`, `tests/test_state_ny.py`

- [ ] Use the Open Legislation API with the Task 0.2 key. Workers' Comp Law article 9 (PFL).
- [ ] Key read from env, never hardcoded.
- [ ] Commit.

## Task 3.3: Ohio, and encoding absence

**Files:** Create `ingest/state_oh.py`, `corpus/absence/oh.yaml`

- [ ] Parse `codes.ohio.gov`. Extract `Effective: <date>` into `effective_from`.
- [ ] Ingest what Ohio does have: military family leave, jury duty, voting leave.
- [ ] **Step 3: Encode the absence explicitly.**

Ohio has no state family-leave statute for private employers. A retrieval miss and a genuine absence must not look the same to the agent:

```yaml
- jurisdiction: OH
  topic: family_medical_leave
  finding: no_state_provision
  effect: federal_controls
  verified_on: 2026-08-25
  note: >
    Ohio has no state FMLA equivalent for private employers.
    Absence is a fact about the corpus, not a retrieval failure.
```

- [ ] **Step 4: Test that absence is distinguishable**

```python
def test_absence_is_not_a_retrieval_miss():
    r = lookup_state_provision("OH", "family_medical_leave")
    assert r.finding == "no_state_provision"
    assert r is not None  # absence is a record, never None
```

- [ ] Commit.

---

# PHASE 4: Chunking, embedding, vector store

## Task 4.1: Structure-aware chunking

**Files:** Create `retrieval/chunking.py`, `tests/test_chunking.py`

- [ ] Test that a section is never split mid-subsection and `section_path` survives on every chunk.
- [ ] Implement structure-aware splitting. Also implement `fixed_size_chunker` as the baseline to beat.
- [ ] Both must satisfy the same interface so the eval can swap them.
- [ ] Commit.

## Task 4.2: Embedding, as decision log entry #1

**Files:** Create `retrieval/embed.py`, `eval/decision_log.md`

- [ ] Implement a provider interface with two backends: a general-purpose hosted model and a legal-domain model.
- [ ] **Do not pick one yet.** Both get measured in Task 5.3.
- [ ] Open `eval/decision_log.md` with the hypothesis, before the result is known:

```markdown
## D1: Which embedding model for a regulatory corpus?
**Hypothesis.** A legal-domain embedding model beats a general-purpose one on
retrieval recall for statutory text, because the corpus is dense with terms of art.
**Metric.** recall@10 on the full scenario set.
**Result.** _pending Task 5.3_
```

- [ ] Commit.

## Task 4.3: Qdrant collection

**Files:** Create `retrieval/store.py`, `tests/test_store.py`

- [ ] Test against a local Qdrant from `docker-compose.yml`.
- [ ] **Test the filter is hard, not a ranking hint:**

```python
def test_jurisdiction_filter_excludes_other_states():
    hits = search("parental leave", jurisdiction="OH", k=20)
    assert all(h.payload["jurisdiction"] in ("OH", "US") for h in hits)
```

- [ ] Named vectors for dense plus sparse hybrid. Payload indexes on `jurisdiction`, `authority_layer`, `effective_from`, `effective_to`.
- [ ] Commit.

---

# PHASE 5: Baseline and eval harness

**The baseline is built before the agent.** Without a measured baseline the decision log has nothing to compare against, and the demo's side-by-side toggle has nothing to show.

## Task 5.1: Naive RAG baseline

**Files:** Create `agent/baseline.py`

- [ ] Deliberately naive: top-k retrieval, no jurisdiction filter, no precedence, no date filter, single prompt.
- [ ] **This is not a strawman, it is what most implementations actually do.** It must be a fair, competent version of the naive approach.
- [ ] Commit.

## Task 5.2: Eval harness

**Files:** Create `eval/run.py`, `eval/metrics/*.py`

- [ ] Implement each metric from the spec as its own scorer.
- [ ] **Groundedness is scored by a different model than the one that generated the answer.** Self-grading is not evaluation.
- [ ] Emit a JSON report plus a Markdown summary per run, with the git SHA.
- [ ] Commit.

## Task 5.3: Measure, and close D1

- [ ] Run the full scenario set against the baseline with both embedding models and both chunkers.
- [ ] **Record real numbers in `eval/decision_log.md`.** Whatever they are. If the legal-domain model loses, that entry is more valuable, not less.
- [ ] Commit the report as the baseline all later work is measured against.

---

# PHASE 6 onward: outline

Phases 0 to 5 are specified at task granularity because they are the critical path to a measured baseline. The phases below are scoped but **deliberately not expanded into steps yet**, because their design should be informed by what Phase 5 measures. Expand each when reached.

**Phase 6: Agent graph.** LangGraph nodes per the spec. One task per node, each with scenario-slice tests. `resolve` implements the four precedence rules and emits a structured trace. `verify` uses a different model family than `compose`.

**Phase 7: Observability.** Langfuse spans per node, recording retrieval filters, precedence decision, per-stage cost and latency.

**Phase 8: API and protection.** FastAPI. Per-IP limit, per-session daily quota, global daily circuit breaker, input length cap, pre-computed responses for the six curated scenarios.

**Phase 9: Demo UI.** Six scenario buttons, baseline toggle, live trace panel, free text input. The supersession scenario runs one question at two dates side by side.

**Phase 10: Ship.** Qdrant Cloud, Langfuse Cloud, app on Fly.io. README leads with the decision log, setup last.

---

## Risks

- **Scenario authoring is the schedule.** Roughly 90 hand-written cases with correct citations. Write them in slices, commit each slice.
- **Ingestion sprawl.** Hard rule: ingestion stops when the scenario set is answerable. Not when the corpus feels complete.
- **State HTML will break.** These are scraped pages, not APIs. Every adapter is fixture-tested so a site change fails a test rather than silently poisoning the corpus.
- **Precedence rules oversimplify real law.** They do. Hence the disclaimer and the leave-only scope.
