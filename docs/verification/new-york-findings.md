# New York: the cited section does not support the claim

*2026-08-26, read-only against the ingested fixture while the DL-1 run was in
flight.*

## What two scenarios assert

`conflict-006` and `ambiguous-015` both turn on New York keying Paid Family Leave
eligibility on **weeks worked**, rather than on the federal shape of months of
service plus hours. `ambiguous-015` goes further and makes `weeks_worked_12mo`
the specific missing fact a clarifying question must ask for. Both cite
`N.Y. Workers' Comp. Law 204`.

`weeks_worked_12mo` was added to the schema in Phase 3.5 specifically to express
this, on the grounds that the eligibility test could not otherwise be represented.

## What section 204 actually says

Nothing about eligibility. It governs **benefit amounts and duration**:

> ...on or after January first, two thousand eighteen shall not exceed eight
> weeks during any fifty-two week calendar period and shall be fifty percent of
> the employee's average weekly wage... on or after January first of each
> succeeding year, shall not exceed twelve weeks... and shall be sixty-seven
> percent...

It refers to an "eligible employee" twice and defines the term nowhere. Searching
the ingested text: no "twenty-six", no "26 consecutive weeks", no "175 days", no
weeks-based eligibility clause of any kind.

**The section is real, the claim about New York may well be correct, and the
citation does not support it.** Eligibility is defined elsewhere in the article,
almost certainly `WCL 201`, which was never ingested because no scenario appeared
to need it.

## Why this slipped through

The Phase 3 scope was read off the scenario set: New York required exactly one
citation, so exactly one section was fetched. That rule is sound and worked
everywhere else. It fails when **the scenario cites the wrong section**, because
then the corpus faithfully ingests the wrong thing and every later check passes.

`ingest/state_ny.py` parsed 204 correctly. The loader confirmed the citation
resolves. Retrieval returns it. Nothing was broken except the claim.

## The fix, once the evaluation completes

Two options, and they are not equivalent.

**Ingest `WCL 201` and repoint the eligibility scenarios at it.** Correct, and it
grows the New York layer by one section for two scenarios.

**Or rewrite the scenarios around what 204 does support**, which is the benefit
schedule: how many weeks of paid family leave, at what percentage of average
weekly wage. That is a genuine New York distinctive and needs no new ingestion.

The first is more honest to the original intent. The second is cheaper and still
tests jurisdiction handling. **Deciding this needs a look at 201's actual text**
rather than a preference, because if 201 does not state a weeks-worked test
either, the original claim was wrong about New York law and not merely
misattributed.

## Standing

Both scenarios remain `verified: false`, correctly. Neither should be scored
until the citation matches the claim.

---

## Resolved: misattributed, not wrong

`WCL 203`, titled "Employees eligible for benefits under section two hundred four
of this article", states that employees "in employment of a covered employer for
**twenty-six or more consecutive weeks**... shall be eligible for **family leave
benefits** as provided in section two hundred four of this article."

So the two possibilities collapse to the better one. **New York does key Paid
Family Leave eligibility on weeks of employment**, exactly as the scenarios
claim. The section cited was simply the wrong one: 203 sets eligibility and 204
sets the benefit it points to.

This also vindicates the `weeks_worked_12mo` field added in Phase 3.5. The test
it represents is real; the corpus just never contained the provision stating it.

Checked along the way, and worth recording so nobody re-treads it: `WCL 201`
(Definitions, 24,000 characters) does not define "eligible employee" at all, and
`WCL 202` (Covered employer) carries no eligibility terms. The eligibility rule
is only in 203.

### The fix

Ingest `WCL 203` and repoint `conflict-006` and `ambiguous-015` at it. No
scenario needs rewriting, and the New York layer grows by one section for a
reason the eval can point at.

`WCL 203` reports `activeDate` 2026-02-27, which the adapter reads directly, so
no commencement date has to be inferred.

### What this says about the scoping rule

Phase 3 scoped ingestion by reading citations off the scenario set, and DL-11
credited that rule with eliminating an entire Ohio adapter. It is still the right
rule. But it inherits the accuracy of the citations it reads, and a scenario
citing the wrong section produces a corpus that is faithfully, verifiably wrong.

Every automated check passed throughout: the parser handled 204 correctly, the
loader confirmed the citation resolved, retrieval returned it. **Only reading the
statute against the claim caught it.** That is the same conclusion as the Ohio
records, arrived at from the opposite direction: there, a claim about absent law;
here, a claim about present law pointed at the wrong provision.
