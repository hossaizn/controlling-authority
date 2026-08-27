"""Tests for the resolve node: evidence extraction, and what it hands precedence.

The rules themselves are tested in `test_precedence.py`. These cover the seam,
which is where a model's output meets code that assumes things about it.
"""

from __future__ import annotations

from datetime import date

from agent.nodes.resolve import (
    LAYERS,
    cap_per_layer,
    PROMPT_VERSION,
    READ_TOOL,
    _passages_by_layer,
    _to_findings,
    make_resolve,
    resolve_citation,
)
from agent.state import initial_state
from domain import EmployeeContext
from retrieval.store import SearchHit


class FakeCaller:
    def __init__(self, findings):
        self.result = {"findings": findings}
        self.system = self.user = None

    def call(self, system, user, tool, model, **kw):
        self.system, self.user = system, user
        return self.result


def hit(citation, layer, text="body", status="substantive", heading="h"):
    return SearchHit(
        chunk_id=f"c-{citation}",
        citation=citation,
        authority_layer=layer,
        jurisdiction="US",
        content_status=status,
        heading=heading,
        text=text,
        score=0.5,
    )


def findings_from(raw, valid=None):
    """`_to_findings` validates citations against what was actually retrieved.
    These cases are about normalising layers, so the citations they use are
    treated as retrieved unless a test says otherwise."""
    cites = valid if valid is not None else {r["citation"] for r in raw if r.get("citation")}
    return _to_findings(raw, set(cites))


def run(findings, hits=None, question="am I eligible?"):
    state = initial_state(
        question, employee_context=EmployeeContext(state="CA"), as_of=date(2026, 4, 1)
    )
    if hits is None:
        # The node validates citations against what was actually retrieved, so a
        # fixture that reports a citation must also have retrieved it. Deriving
        # the hits from the findings keeps the two from drifting apart, which is
        # what made three of these fail the moment the validator was added.
        cited = [(f["citation"], f["layer"]) for f in findings if f.get("citation")]
        hits = [hit(c, layer) for c, layer in cited] or [hit("29 CFR 825.110", "federal")]
    state["retrieved"] = hits
    caller = FakeCaller(findings)
    return make_resolve(caller)(state), caller


# --- normalising what the model returned ------------------------------------


def test_a_layer_the_model_omitted_is_silent_not_missing() -> None:
    """Dropping it would make the resolution look like it considered two layers
    when it considered three, and the trace has to show what was rejected."""
    findings = findings_from([
        {"layer": "federal", "outcome": "grants", "citation": "29 CFR 825.110",
         "says": "12 months", "generosity_rank": 1}
    ])
    assert [f.layer for f in findings] == list(LAYERS)
    assert {f.layer for f in findings if f.outcome == "silent"} == {"state", "company"}


def test_a_layer_that_speaks_without_a_citation_is_downgraded_to_silent() -> None:
    """An uncited assertion has nothing behind it, and letting it control would
    put an entitlement on the record with no provision supporting it."""
    findings = findings_from([
        {"layer": "company", "outcome": "grants", "citation": "",
         "says": "we give 10 days", "generosity_rank": 1}
    ])
    company = next(f for f in findings if f.layer == "company")
    assert company.outcome == "silent"
    assert company.speaks_to_question is False


def test_a_silent_layer_carries_no_generosity_rank() -> None:
    findings = findings_from([
        {"layer": "state", "outcome": "silent", "citation": "", "says": "",
         "generosity_rank": 0}
    ])
    assert all(f.generosity_rank is None for f in findings)


def test_a_duplicated_layer_keeps_only_the_first() -> None:
    """Precedence raises on duplicates, so they are collapsed here rather than
    reaching it. Two findings for one layer is a malformed answer, not two
    provisions."""
    findings = findings_from([
        {"layer": "state", "outcome": "grants", "citation": "A", "says": "",
         "generosity_rank": 1},
        {"layer": "state", "outcome": "denies", "citation": "B", "says": "",
         "generosity_rank": 2},
    ])
    state = next(f for f in findings if f.layer == "state")
    assert state.citation == "A"
    assert len([f for f in findings if f.layer == "state"]) == 1


def test_a_speaking_layer_carrying_the_not_applicable_rank_is_silent() -> None:
    """`rank or 1` promoted the 0 sentinel to most-generous. A layer that claims
    to speak while carrying no usable rank is malformed evidence, and if it also
    denies, the integrity check raises and the whole answer degrades to a
    referral on the strength of a coercion."""
    findings = findings_from([
        {"layer": "federal", "outcome": "denies", "citation": "29 CFR 825.122",
         "says": "not covered", "generosity_rank": 0},
        {"layer": "company", "outcome": "grants", "citation": "LEAVE-001",
         "says": "covered", "generosity_rank": 1},
    ])
    federal = next(f for f in findings if f.layer == "federal")
    assert federal.outcome == "silent"
    assert federal.generosity_rank is None


def test_a_negative_rank_is_also_treated_as_silent() -> None:
    findings = findings_from([
        {"layer": "state", "outcome": "grants", "citation": "A", "says": "",
         "generosity_rank": -1}
    ])
    assert all(f.outcome == "silent" for f in findings)


def test_an_unrecognised_outcome_is_treated_as_silent() -> None:
    findings = findings_from([
        {"layer": "federal", "outcome": "maybe", "citation": "X", "says": "",
         "generosity_rank": 1}
    ])
    assert all(f.outcome == "silent" for f in findings)


# --- citations are validated against what was actually retrieved ------------


def test_a_whole_passage_pasted_into_the_citation_field_is_recovered() -> None:
    """The defect this validator exists for, found by `must_address` scoring
    reading 0/8 while every resolution looked correct.

    Told to copy the citation "exactly as it appears in the passages", and given
    passages rendered as `[citation] heading / body`, the model copied the whole
    block.
    """
    pasted = (
        "[LEAVE-002] Parental Leave... Employees who have completed 18 months of "
        "continuous service at the time leave begins are eligible."
    )
    assert resolve_citation(pasted, {"LEAVE-002", "29 CFR 825.120"}) == "LEAVE-002"


def test_the_longest_matching_citation_wins() -> None:
    """`Cal. Gov. Code 12945` is a prefix of `Cal. Gov. Code 12945.2` and both
    are in this corpus. DL-12 records the same trap producing the right citation
    attached to the wrong statute."""
    valid = {"Cal. Gov. Code 12945", "Cal. Gov. Code 12945.2"}
    assert resolve_citation("[Cal. Gov. Code 12945.2] text", valid) == "Cal. Gov. Code 12945.2"


def test_an_exact_citation_passes_through() -> None:
    assert resolve_citation("29 CFR 825.200", {"29 CFR 825.200"}) == "29 CFR 825.200"


def test_a_citation_that_was_never_retrieved_is_rejected() -> None:
    """No amount of prompting reliably stops a model citing something it was not
    given, and it cannot be trusted to police itself. So the check is code."""
    assert resolve_citation("29 CFR 825.999", {"29 CFR 825.200"}) is None


def test_a_rejected_citation_makes_the_layer_silent_rather_than_controlling() -> None:
    """An entitlement resting on a provision nobody retrieved must not reach the
    answer as though it were supported."""
    findings = findings_from(
        [{"layer": "federal", "outcome": "grants", "citation": "29 CFR 825.999",
          "says": "invented", "generosity_rank": 1}],
        valid={"29 CFR 825.200"},
    )
    assert all(f.outcome == "silent" for f in findings)


def test_an_empty_citation_resolves_to_nothing() -> None:
    assert resolve_citation("", {"29 CFR 825.200"}) is None


# --- the prompt gets the evidence it needs ----------------------------------


def test_passages_are_grouped_by_layer_and_labelled() -> None:
    """The layer is metadata we already hold. Making a model infer it from a
    citation string invites an error precedence cannot detect."""
    _, caller = run(
        [], hits=[hit("29 CFR 825.200", "federal"), hit("LEAVE-001", "company")]
    )
    assert "### FEDERAL" in caller.user
    assert "### COMPANY" in caller.user
    assert caller.user.index("### FEDERAL") < caller.user.index("### COMPANY")


def test_a_layer_with_no_passages_says_so_explicitly() -> None:
    _, caller = run([], hits=[hit("LEAVE-001", "company")])
    assert "no passages retrieved" in caller.user


def test_an_absence_record_is_marked_as_a_recorded_silence() -> None:
    """A retrieval miss and a genuine absence demand opposite responses and must
    never look alike."""
    _, caller = run([], hits=[hit("OH-absent", "state", status="absent")])
    assert "record that the law is silent" in caller.user


def test_the_model_is_not_asked_which_layer_controls() -> None:
    """The split the whole module exists for."""
    _, caller = run([])
    assert "NOT deciding which authority wins" in caller.system
    fields = READ_TOOL["input_schema"]["properties"]["findings"]["items"]["properties"]
    assert "controlling" not in fields
    assert set(fields) == {"layer", "outcome", "citation", "says", "generosity_rank"}


def test_the_prompt_version_is_in_the_cache_key_material() -> None:
    _, caller = run([])
    assert PROMPT_VERSION in caller.system


def test_grouping_puts_each_hit_under_its_own_layer() -> None:
    grouped = _passages_by_layer(
        {"retrieved": [hit("a", "federal"), hit("b", "company"), hit("c", "federal")]}
    )
    assert [h.citation for h in grouped["federal"]] == ["a", "c"]
    assert [h.citation for h in grouped["company"]] == ["b"]


# --- end to end through the node --------------------------------------------


def test_the_resolution_reaches_the_state_with_the_rule_that_fired() -> None:
    out, _ = run([
        {"layer": "state", "outcome": "grants", "citation": "Cal. Gov. Code 12945.2",
         "says": "12 months", "generosity_rank": 1},
        {"layer": "company", "outcome": "denies", "citation": "LEAVE-002",
         "says": "18 months", "generosity_rank": 2},
    ])
    assert out["resolution"].controlling == "state"
    assert out["resolution"].rule == "policy_below_floor"


def test_inconsistent_evidence_degrades_rather_than_resolving_on_a_guess() -> None:
    """A layer that denies ranked above one that grants is contradictory. A
    resolution computed from it would still look like a resolution."""
    out, _ = run([
        {"layer": "federal", "outcome": "denies", "citation": "29 CFR 825.122",
         "says": "not covered", "generosity_rank": 1},
        {"layer": "company", "outcome": "grants", "citation": "LEAVE-001",
         "says": "covered", "generosity_rank": 2},
    ])
    assert out["resolution"].controlling is None
    assert out["resolution"].rule == "not_reached"
    assert "inconsistent" in out["trace"][0].summary


def test_nothing_retrieved_resolves_to_nothing_without_calling_a_model() -> None:
    state = initial_state("q", as_of=date(2026, 4, 1))
    state["retrieved"] = []

    class Exploding:
        def call(self, *a, **k):
            raise AssertionError("a model was called with no evidence to read")

    out = make_resolve(Exploding())(state)
    assert out["resolution"].controlling is None


def test_the_trace_records_every_layer_considered_not_just_the_winner() -> None:
    out, _ = run([
        {"layer": "state", "outcome": "grants", "citation": "Cal. Gov. Code 12945.2",
         "says": "grants", "generosity_rank": 1},
    ])
    considered = out["trace"][0].detail["considered"]
    assert {c["layer"] for c in considered} == set(LAYERS)


def test_the_summary_explains_the_rule_in_plain_words() -> None:
    """The trace is a feature, not a debug flag, and this line is the one that
    explains the whole product to a non-technical reader."""
    out, _ = run([
        {"layer": "state", "outcome": "grants", "citation": "Cal. Lab. Code 246",
         "says": "same accrual", "generosity_rank": 1},
        {"layer": "company", "outcome": "grants", "citation": "LEAVE-004-v2",
         "says": "same accrual", "generosity_rank": 1},
    ])
    assert "restates the statute" in out["trace"][0].summary


# --- the per-layer passage cap (DL-38) --------------------------------------


def test_no_cap_keeps_every_passage() -> None:
    """`None` is the behaviour this node had before the question was asked."""
    hits = [hit(f"fed-{i}", "federal") for i in range(6)]
    assert len(cap_per_layer(hits, None)) == 6


def test_the_cap_counts_per_layer_not_overall() -> None:
    """A global cut of 2 would leave two federal passages and nothing else,
    and precedence needs a finding from each layer to compare anything."""
    hits = (
        [hit(f"fed-{i}", "federal") for i in range(5)]
        + [hit("ca-1", "state")]
        + [hit("hb-1", "company")]
    )
    kept = cap_per_layer(hits, 2)
    assert [h.citation for h in kept] == ["fed-0", "fed-1", "ca-1", "hb-1"]


def test_the_cap_keeps_the_top_ranked_passages_of_each_layer() -> None:
    """Retrieval order is the signal: DL-38 measured 84% of cited evidence as
    the top passage of its layer. Keeping a later one would discard that."""
    hits = [hit(f"fed-{i}", "federal") for i in range(4)]
    assert [h.citation for h in cap_per_layer(hits, 2)] == ["fed-0", "fed-1"]


def test_a_layer_thinner_than_the_cap_is_untouched() -> None:
    hits = [hit("ca-1", "state"), hit("hb-1", "company")]
    assert len(cap_per_layer(hits, 3)) == 2


def test_a_cap_of_zero_shows_nothing() -> None:
    """Distinct from None. `cap or default` would silently turn this into the
    uncapped case, which is the opposite of what it asks for."""
    assert cap_per_layer([hit("fed-0", "federal")], 0) == []


def test_the_cap_reaches_the_prompt() -> None:
    hits = [hit(f"fed-{i}", "federal", text=f"body-{i}") for i in range(4)]
    state = initial_state(
        "q", employee_context=EmployeeContext(state="CA"), as_of=date(2026, 4, 1)
    )
    state["retrieved"] = hits
    caller = FakeCaller([])
    make_resolve(caller, passage_cap=2)(state)
    assert "body-1" in caller.user
    assert "body-2" not in caller.user, "passage past the cap reached the prompt"


def test_a_citation_beyond_the_cap_is_not_accepted() -> None:
    """The model cannot have read it. Accepting it would let `verify` check the
    answer against evidence that never entered the decision."""
    hits = [hit(f"fed-{i}", "federal") for i in range(4)]
    state = initial_state(
        "q", employee_context=EmployeeContext(state="CA"), as_of=date(2026, 4, 1)
    )
    state["retrieved"] = hits
    caller = FakeCaller([
        {"layer": "federal", "outcome": "grants", "citation": "fed-3",
         "says": "x", "generosity_rank": 1}
    ])
    out = make_resolve(caller, passage_cap=2)(state)
    federal = next(f for f in out["resolution"].considered if f.layer == "federal")
    assert federal.citation is None
    assert federal.outcome == "silent"


def test_the_trace_records_what_the_decision_was_allowed_to_see() -> None:
    """A capped run and an uncapped one must not look alike after the fact."""
    hits = [hit(f"fed-{i}", "federal") for i in range(5)]
    state = initial_state(
        "q", employee_context=EmployeeContext(state="CA"), as_of=date(2026, 4, 1)
    )
    state["retrieved"] = hits
    out = make_resolve(FakeCaller([]), passage_cap=2)(state)
    detail = out["trace"][0].detail
    assert detail["passage_cap"] == 2
    assert detail["passages_shown"] == 2
    assert detail["passages_retrieved"] == 5
