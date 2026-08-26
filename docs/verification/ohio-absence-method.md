# Verifying an absence: method and findings

*Drafted 2026-08-26 while the DL-1 run was in flight. No corpus or scenario file
was touched, so the measurement in progress is unaffected.*

## Why this needs a method at all

Every other claim in this project is positive. The statute says "at least 12
months", the ingested text contains that string, the claim is verified. Grep
settles it.

**The Ohio records claim a negative**: that no Ohio provision requires paid sick
leave, restricts vacation forfeiture, or mandates jury duty pay. You cannot grep
for the absence of a law. Nothing in the corpus can confirm it, because the
corpus is the thing whose completeness is in question.

So these claims can never reach the same standard as the others, and pretending
otherwise by ticking the same kind of box would be the more dishonest option.
What follows is a documented, repeatable search that came up mostly empty, plus
the one place it did not.

## The search

**1. Enumerate the whole of Title 41, Labor and Industry.** If Ohio imposed a
general leave obligation on private employers, its chapter would be here.

Twenty-seven chapters. Not one concerns family, medical, sick, parental or
bereavement leave. The nearest candidates are 4111 Minimum Fair Wage Standards,
4112 Civil Rights Commission, 4113 Miscellaneous Labor Provisions, and 4115
Wages and Hours on Public Works, which applies to public works only.

**2. Search every section title in the candidate chapters** for leave, pregnancy,
maternity, family, sick, vacation, bereavement, jury, witness, military, absence.

- **4112**, 31 sections, one hit: *Inherently military civilian job
  discrimination claims*. Not a leave entitlement.
- **4113**, 37 sections, two hits: *Leave of absence for union management
  relations* and *Absence by volunteer firefighter or emergency medical services
  provider*. Both narrow, neither general.

**3. Search the Administrative Code, not only the Revised Code.** This step
exists because step 2 nearly produced a false clearance, and it is the reason the
method has three steps rather than two.

## What the search found: one absence record is wrong

**Ohio Administrative Code Rule 4112-5-05(G) creates a maternity leave
obligation.** Subsection (G)(2):

> Where termination of employment of an employee who is temporarily disabled due
> to pregnancy or a related medical condition is caused by an employment policy
> under which insufficient or no maternity leave is available, such termination
> shall constitute unlawful sex discrimination.

The `parental_leave` record currently states that Ohio has "no state parental or
bonding leave entitlement for employees of private employers". **That is
overbroad and false as written.** Ohio imposes an obligation regarding leave for
pregnancy-related disability; it arises under the Civil Rights Act by
administrative rule rather than by statute, which is exactly why an enumeration
of statute titles missed it.

The record has to be narrowed to what remains true: Ohio has no general
*bonding* leave entitlement, and no paid family leave benefit, while pregnancy
disability is separately protected.

**This is the finding that justifies the whole exercise.** The claim was written
from recall in Phase 3, survived a review in Phase 3.5, and is false. Had it been
ticked on the strength of "Title 41 contains no leave chapter", the corpus would
have asserted something untrue about the law and every Ohio parental-leave
scenario would have rested on it.

## Status these records can honestly reach

`verified_on` is recorded together with the method, never as a bare date. The
distinction stated in the record itself:

- **Positive claims** (a rule exists and says X) reach the same standard as the
  rest of the corpus, since the text can be quoted.
- **Negative claims** (no provision requires X) reach *searched and not found*,
  scoped to the Revised Code Title 41 and the Administrative Code chapters
  reached above, as of the search date.

The second is weaker and is labelled as such wherever it appears. An absence
record is evidence that a search came up empty, not proof that nothing exists.

## Still to do, after the DL-1 run completes

- Narrow the `parental_leave` record and record the OAC rule.
- Run steps 1 to 3 for the remaining seven topics rather than assuming this one
  was the only mistake.
- Ingest `Cal. Lab. Code 246` and California's bereavement provision, which four
  scenarios depend on and neither is in the corpus.

---

## Second pass: two more findings, and a correction to the method itself

Run while the DL-1 evaluation was in flight. Still read-only; no corpus or
scenario file touched.

### The method was incomplete: a single-title sweep is not enough

Step 1 enumerated Title 41, Labor and Industry, on the reasoning that a
private-employer leave obligation would live there. **Ohio's jury duty
protection is `ORC 2313.19`, in Title 23, Courts.** A Title 41 sweep misses it
entirely.

The `jury_duty_pay` record survives only because it was written narrowly in
Phase 3.5: it claims no statute requires *payment* and explicitly disclaims any
claim about protection from discharge. `2313.19` bars discharge and says nothing
about pay, so the record holds. Written as "Ohio has no jury duty statute", it
would have been false.

**Two structural lessons.** Search across titles, not within the one that seems
topical. And keep every negative claim as narrow as the scenario needs, because
narrow claims survive contact with statutes you did not think to look for.

### `vacation_forfeiture` is wrong as written

`ORC 4113.15(D)(2)` defines "fringe benefits" for wage-payment purposes as
including "vacation, separation, or holiday pay". Subsection (C) provides that an
employer party to an agreement to pay fringe benefits **becomes a trustee** of
the funds that agreement requires to be paid.

The record currently states that Ohio "does not classify accrued vacation as
wages that cannot be forfeited". The first half of that is false: Ohio's
wage-payment statute expressly folds vacation pay into the fringe benefits it
governs.

**The conclusion still holds, for a different reason than the record gives.**
Ohio enforces the *agreement*. Where a policy provides that days above a cap are
forfeited, no obligation to pay them arises, so nothing is held in trust and
nothing is owed. California is the opposite: `Cal. Lab. Code 227.3` voids the
forfeiture term itself.

That distinction is sharper than the original claim and makes `conflict-008` a
better scenario. Ohio does not ignore vacation; it enforces what the handbook
says about it, which is exactly why the handbook controls there and does not
control in California.

**Rewrite required**, from "Ohio does not classify vacation as wages" to "Ohio
does not prohibit forfeiture; its wage-payment statute enforces the terms the
policy itself sets."

### Running total

Two of eight absence records were false as written, both drafted from recall in
Phase 3 and both surviving a full review in Phase 3.5. Neither would have failed
a plausibility check. The remaining topics still need the same treatment, now
across titles and including the Administrative Code.

### `military_leave` is wrong too, and in a way worth naming precisely

`ORC 5903.02(B)`, in Chapter 5903 (Soldiers, Sailors, Marines), provides that any
person whose absence from employment is necessitated by service in the uniformed
services "has the same reinstatement and reemployment rights in this state that a
person has under" USERRA. It names no employer category, so it is not confined to
public employment.

The record states that "Ohio has no statute governing military leave for
employees of private employers that adds to this corpus". The qualifier does some
work, but the leading clause is false: **`5903.02` is exactly such a statute.**

What is true is narrower and more interesting. Ohio's provision *mirrors* the
federal floor rather than exceeding it, so it changes no outcome in this corpus.
An absence record was the wrong instrument: this is not silence, it is a state
law that deliberately tracks federal law. The corpus should say so, because
"there is no Ohio rule" and "Ohio's rule is the federal rule restated" produce
the same answer for opposite reasons, and only one of them is true.

This is also the **third** chapter found outside Title 41 (after Title 23 for
jury duty and the Administrative Code for maternity). The single-title sweep was
not a small oversight; it was the wrong shape of search.

### Running total after the second pass

| record | status |
|---|---|
| `parental_leave` | **false** — OAC 4112-5-05(G) requires maternity leave |
| `vacation_forfeiture` | **false as written** — ORC 4113.15(D)(2) classifies vacation pay as a fringe benefit |
| `military_leave` | **false as written** — ORC 5903.02 grants USERRA-equivalent rights under Ohio law |
| `jury_duty_pay` | holds, because it was scoped to payment only |
| `family_medical_leave` | no contrary provision found yet |
| `paid_sick_leave` | not yet searched across titles |
| `bereavement_leave` | not yet searched across titles |
| `witness_duty_pay` | not yet searched across titles |

**Three of eight are wrong**, all written from recall in Phase 3, all through a
review in Phase 3.5. The three that survive so far do so because they were
scoped narrowly rather than because they were better researched.

### A limitation the method cannot currently remove

The correct fix for a title-by-title sweep is a keyword search across the whole
code. `codes.ohio.gov` exposes no search endpoint reachable programmatically:
requests to it return an HTTP error, and the site's own search appears to run
client-side.

So the method remains: enumerate candidate chapters across **multiple titles**,
plus the Administrative Code, and read section titles. That is better than the
single-title version it replaces and is still not exhaustive. Every finding so
far came from guessing the right chapter, and the fourth contrary provision may
sit in a chapter nobody guessed.

**This is why the remaining four records are marked "searched, not found, scope
stated" rather than verified.** They are the same standard as the three that were
wrong before anyone looked.
