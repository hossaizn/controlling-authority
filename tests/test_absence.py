"""Absence records must be distinguishable from a retrieval failure."""

from __future__ import annotations

from datetime import date

import pytest

from ingest.absence import load_absence_records

OBSERVED = date(2026, 8, 26)


@pytest.fixture(scope="module")
def ohio():
    return load_absence_records("OH", observed_on=OBSERVED)


def test_every_topic_the_scenarios_touch_is_covered(ohio) -> None:
    topics = {d.doc_id.split("absence-")[1] for d in ohio}
    assert topics == {
        "family_medical_leave",
        "parental_leave",
        "paid_sick_leave",
        "vacation_forfeiture",
        "bereavement_leave",
        "jury_duty_pay",
        "witness_duty_pay",
        "military_leave",
    }


def test_absences_carry_retrievable_text(ohio) -> None:
    """The whole point. An empty result is indistinguishable from a miss."""
    assert all(d.text.strip() for d in ohio)
    fml = next(d for d in ohio if d.doc_id.endswith("family_medical_leave"))
    assert "no state family and medical leave statute" in fml.text


def test_absences_are_marked_absent_not_substantive(ohio) -> None:
    """So they can never be cited as though they were a statute."""
    assert {d.content_status for d in ohio} == {"absent"}


def test_citation_does_not_look_like_a_statute(ohio) -> None:
    for d in ohio:
        assert "no state provision" in d.citation
        assert "Ohio Rev. Code" not in d.citation


def test_absences_are_in_force_for_any_query_the_corpus_answers(ohio) -> None:
    """A standing fact about a body of law, not a dated provision.

    in_force_on short-circuits on content_status rather than comparing the
    sentinel date, so the 1900 placeholder can never leak into an answer or be
    read as a commencement date.
    """
    d = ohio[0]
    assert d.in_force_on(date(2023, 6, 15))
    assert d.in_force_on(date(2026, 8, 26))
    assert d.in_force_on(date(1850, 1, 1))


def test_absences_are_unverified_until_checked() -> None:
    """DL-3 applies to these exactly as it does to statutory claims.

    Checked against a parsed field. The first version substring-matched
    "verified_on=None" in source_note, which a YAML value of the *string*
    "null" satisfies just as well as a real null, silently disabling the guard
    that depends on it.
    """
    from ingest.absence import load_absence_index

    # Written with an exit condition. The first version asserted every record
    # was unverified, which had to fail the moment verification happened: the
    # DL-7 anti-pattern of a test that breaks when the project succeeds.
    # What matters is that the flag holds a real date or a real null, never a
    # placeholder string, and that pending records are still visible as pending.
    for r in load_absence_index("OH"):
        assert r.verified_on is None or r.verified_on.year >= 2026


def test_a_string_null_is_rejected_not_treated_as_unverified() -> None:
    """The exact hole review found: quoting the value in YAML would have read
    as verified-looking text while meaning nothing."""
    import tempfile
    from pathlib import Path as _Path

    import ingest.absence as absence_mod

    with tempfile.TemporaryDirectory() as tmp:
        (_Path(tmp) / "zz.yaml").write_text(
            "- topic: t\n  effect: federal_controls\n  text: x\n  verified_on: \"null\"\n"
        )
        original = absence_mod.ABSENCE_DIR
        absence_mod.ABSENCE_DIR = _Path(tmp)
        try:
            with pytest.raises(ValueError, match="verified_on must be a date"):
                absence_mod.load_absence_index("ZZ")
        finally:
            absence_mod.ABSENCE_DIR = original


def test_effect_is_recorded_so_precedence_can_use_it() -> None:
    from ingest.absence import load_absence_index

    by_topic = {r.topic: r for r in load_absence_index("OH")}
    assert by_topic["family_medical_leave"].effect == "federal_controls"
    assert by_topic["vacation_forfeiture"].effect == "employer_policy_controls"


def test_unknown_jurisdiction_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_absence_records("TX")


def test_every_ohio_company_controlled_scenario_has_a_covering_absence() -> None:
    """Under precedence rule 5, "the handbook controls" in Ohio asserts that
    nothing else addresses the topic. That assertion needs a record.

    Review found three scenarios resting on silence nobody had written down.
    """
    from eval.scenarios.loader import load_all

    topics = {d.doc_id.split("absence-")[1] for d in load_absence_records("OH")}
    # Handbook policy -> the absence topic that licenses "company controls".
    covered_by = {
        "LEAVE-003": "bereavement_leave",
        "LEAVE-004-v1": "paid_sick_leave",
        "LEAVE-004-v2": "paid_sick_leave",
        "LEAVE-006": {"jury_duty_pay", "witness_duty_pay"},
        "LEAVE-007": "military_leave",
        "LEAVE-008": "vacation_forfeiture",
    }
    missing = []
    for s in load_all():
        if s.employee_context.state != "OH" or s.expected_authority != "company":
            continue
        for cite in s.required_citations:
            need = covered_by.get(cite)
            if need is None:
                missing.append(f"{s.scenario_id}: {cite} has no mapped absence topic")
            elif isinstance(need, set):
                if not (need & topics):
                    missing.append(f"{s.scenario_id}: none of {need} recorded")
            elif need not in topics:
                missing.append(f"{s.scenario_id}: absence {need!r} not recorded")
    assert not missing, missing
