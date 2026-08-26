# Seeded defects: ground truth

**This file is never ingested and never shown to the agent.** It records which handbook policies carry deliberate faults, and what the correct resolution is.

> **Verification status.** The statutory claims below were drafted from recall and are marked unverified. **Each must be checked against the actual ingested statutory text in Phase 3 before any scenario depending on it is scored.** Ground truth asserted from memory is not ground truth. Where verification contradicts what is written here, this file changes, not the statute.

---

## D-1: Parental leave service requirement is below the California floor

**Policy:** `LEAVE-002` Parental Leave
**Fault:** Requires **18 months** of continuous service.
**Why it is a fault:** California's CFRA sets eligibility at 12 months of service plus 1,250 hours. An employee in California at 14 months with sufficient hours is entitled to bonding leave that this policy denies.

**Correct resolution:** Statute controls. The 18-month term is unenforceable as to California employees. The handbook must be cited as the non-controlling source so the employee understands why the answer differs from what they read.

**Jurisdiction-dependent:** For an Ohio employee, no state provision applies and federal FMLA's 12-month/1,250-hour test governs, so the same fault exists against the federal floor.

- [ ] Verify CFRA eligibility threshold against ingested CA text
- [x] Verify FMLA eligibility threshold against ingested 29 CFR 825 text — **verified 2026-08-25**. `29 CFR 825.110`: "at least 12 months" and "at least 1,250 hours of service". "At least" confirms the inclusive boundary reading in `conflict-017`.

---

## D-2: Bereavement leave exceeds statute

**Policy:** `LEAVE-003` Bereavement Leave
**Fault:** None, deliberately. **10 days paid**, no service requirement.
**Why it is here:** This is the mirror case of D-1 and exists to stop the agent learning "statute always wins."

**Correct resolution:** The handbook controls, because policy may exceed a statutory floor. Answering with the statutory minimum here is wrong.

- [ ] Verify CA bereavement leave entitlement and whether it is paid or unpaid
- [ ] Verify NY and OH have no bereavement provision that exceeds 10 paid days

---

## D-3: Paid sick leave supersession

**Policy:** `LEAVE-004` v1 (through 2023-12-31) and v2 (from 2024-01-01)
**Fault:** Two versions in force at different times. v1 grants 3 days / 24 hours, v2 grants 5 days / 40 hours.

**Correct resolution:** Depends entirely on the query's as-of date. A query dated 2023 must be answered from v1; a query dated 2024 or later from v2. Citing the superseded version for a current query is a `forbidden_citation`, and vice versa.

**Why this pairing:** California raised its paid sick leave minimum effective 2024-01-01, so the handbook revision tracks a real statutory change rather than an arbitrary one.

- [ ] Verify the CA paid sick leave minimum before and after 2024-01-01
- [ ] Confirm the pre-2024 minimum makes v1 lawful for its period, so the supersession case is about dates and not about a second conflict

---

## D-4: Grandparent care is absent from the handbook

**Policy:** none. `LEAVE-001` lists spouse, child and parent, mirroring federal scope.
**Fault:** The handbook is silent on leave to care for a grandparent.

**Correct resolution, and it splits by jurisdiction:**

- **California:** CFRA's family definition is broader than federal and is understood to include grandparent. State law controls and leave is available, despite handbook silence. This is the "silence is not permission" rule working in the employee's favour.
- **Ohio:** No state provision. Handbook silent. Federal FMLA's care-leave family definition (`29 CFR 825.122`) covers spouse, parent, and son or daughter, and expressly excludes parents-in-law; grandparent is not among them. The correct answer is that no entitlement exists for this purpose, plus a referral. **Not a refusal**, because the system does know the answer: it is "no".

**Corrected twice during verification, and the second correction matters more.**

The original claim, "federal FMLA does not cover grandparents", was too broad: `29 CFR 825.122` mentions grandparents inside the definition of *next of kin of a covered servicemember*, for **military caregiver leave**. A scenario about a wounded servicemember grandparent would have the opposite answer and is not in the set.

The first correction then leaned on the wrong section. It claimed grandparent "appears solely in the military caregiver next-of-kin list", which is also false: the term appears in five ingested sections. **`29 CFR 825.206` states the point outright** — leave "to care for a grandparent" is given as an example of a reason "which does not qualify as FMLA leave" — and `825.701(b)(3)` calls it a purpose not covered by FMLA. The conclusion was right throughout and was resting on the weakest available support: absence from a definition is an inference, while 825.206 is an explicit statement. `conflict-005` now cites 825.206.

The lesson is narrower than "verify claims". It is that a sweep which checks only the section you expected to be dispositive will confirm your conclusion while missing that the corpus supports it better somewhere else.

**Why this is the most valuable case in the set:** the same question has opposite correct answers in two states, and neither answer is retrievable from the most semantically relevant document.

- [ ] Verify CFRA covered family members include grandparent
- [x] Verify federal FMLA excludes grandparent care — **verified 2026-08-25**, corrected 2026-08-26. `29 CFR 825.206` states it explicitly. `825.122` covers spouse, parent, son or daughter for care leave, and mentions grandparent only in the military caregiver next-of-kin list. The term occurs in five ingested sections, not one.

---

## D-5: Personal leave applicability is ambiguous

**Policy:** `LEAVE-005` Personal Leave of Absence
**Fault:** "established employees in good standing", "initial period", "reduced schedules" are all undefined. Nothing in the handbook says what an initial period is.

**Correct resolution:** `clarify`. The agent must ask for the fact that would decide it, most likely tenure or schedule status, rather than guessing or refusing. This is the primary test that clarification fires when it should.

**Paired control:** at least one scenario must ask about personal leave in a way where the answer does **not** depend on the ambiguity, and the correct route is `answer`. Without that pair, an agent that always clarifies on this policy scores perfectly.

---

## D-6: PTO forfeiture clause is below the California floor

**Policy:** `LEAVE-008` Paid Time Off
**Fault:** "Up to 5 unused days carry into the following year and must be used by 31 March. Days beyond that are forfeited."

**Why it is a fault:** California treats accrued vacation as earned wages, which cannot be forfeited. A use-it-or-lose-it clause of this kind is unenforceable there. A cap on further accrual is permitted; retroactive forfeiture of already-accrued days is not.

**Found rather than planned.** This clause was written to make the handbook read realistically, and the conflict was noticed afterwards. It is kept because it is the most realistic defect in the set: this is exactly how real multi-state handbooks break, by carrying a term that is lawful in one state and not in another without anyone noticing.

**Correct resolution:** California employee, statute controls and accrued days cannot be forfeited. Ohio and New York employees, the handbook clause stands unless their state provides otherwise.

- [ ] Verify California's treatment of accrued vacation as wages
- [ ] Verify whether NY or OH restrict forfeiture, which would change the paired cases

---

## D-7: FMLA coverage threshold restated as if it were the only one

**Policy:** `LEAVE-001` Family and Medical Leave
**Fault:** States eligibility requires a location where the company employs "50 or more employees within 75 miles", which is the federal coverage threshold presented as the universal rule.

**Why it is a fault:** California's CFRA reaches smaller employers than the federal threshold. An employee at a small California site reads this clause and concludes she is not covered, when state law says otherwise.

**Correct resolution:** For a California employee below the federal threshold, state law controls and the federal threshold is irrelevant rather than decisive. In Ohio the federal threshold is the whole story and the answer is genuinely no.

**Found in review**, not planned. `conflict-010` and `ambiguous-008/009` already depended on it while it was undocumented, which is exactly the gap this file exists to close.

- [ ] Verify the CFRA employer-size threshold against ingested CA text
- [x] Verify the federal 50-employee threshold — **verified 2026-08-25**. `29 CFR 825.104`: "50 or more employees for each working day during each of 20 or more calendar workweeks".
- [x] Verify the 75-mile component — **corrected 2026-08-26**. The phrase "75 miles" does **not** occur in `825.104`; it is in `825.110` and `825.111`. The original tick claimed a "50-employee/75-mile threshold" verified against 825.104, which overstated what that section says. The two-part test is split across sections and `conflict-010/011` must cite accordingly.

---

## D-8: Family definitions and pregnancy leave are federal-shaped throughout

**Policy:** `LEAVE-001`, and `LEAVE-002` by omission
**Fault:** The handbook's covered-relative list is spouse, child and parent, mirroring federal scope. It is silent on grandparent and sibling. `LEAVE-002` also folds all pregnancy-related time into parental leave.

**Why it is a fault:** California's family definition is broader, and California treats pregnancy disability leave as a separate entitlement from bonding leave. An employee reading `LEAVE-002` would conclude that time signed off before a birth consumes her bonding leave, which is both wrong and expensive for her.

**Correct resolution:** State controls in California for grandparent care (`conflict-004`), sibling care (`ambiguous-010`) and pregnancy disability (`conflict-021`). In Ohio, federal scope governs and the answer is no.

**Partly documented before, partly not.** D-4 covered grandparent only. Sibling and pregnancy disability were relied on by scenarios without appearing here.

- [ ] Verify CFRA covered family members include sibling
- [ ] Verify California pregnancy disability leave is separate from bonding leave

---

## Coverage check

| Defect | Tests | Route |
|--------|-------|-------|
| D-1 | Statute overrides handbook | answer |
| D-2 | Handbook overrides statutory floor | answer |
| D-3 | Effective dating and supersession | answer |
| D-4 | Silence, plus jurisdiction-dependent opposite answers | answer |
| D-5 | Clarification fires, and does not over-fire | clarify + answer |
| D-6 | Same clause lawful in one state, not another | answer |
| D-7 | Federal threshold presented as universal | answer |
| D-8 | Federal-shaped family and pregnancy scope | answer |

Refusal and escalation are not exercised by handbook defects. They are covered by the out-of-scope and adversarial scenario slices instead.
