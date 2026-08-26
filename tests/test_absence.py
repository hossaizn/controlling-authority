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
    """A standing fact about a body of law, not a dated provision. A 2023
    question must still learn that Ohio had no such statute."""
    d = ohio[0]
    assert d.in_force_on(date(2023, 6, 15))
    assert d.in_force_on(date(2026, 8, 26))


def test_absences_are_unverified_until_checked(ohio) -> None:
    """DL-3 applies to these exactly as it does to statutory claims."""
    assert all("verified_on=None" in d.source_note for d in ohio)


def test_effect_is_recorded_so_precedence_can_use_it(ohio) -> None:
    fml = next(d for d in ohio if d.doc_id.endswith("family_medical_leave"))
    vac = next(d for d in ohio if d.doc_id.endswith("vacation_forfeiture"))
    assert "effect=federal_controls" in fml.source_note
    assert "effect=employer_policy_controls" in vac.source_note


def test_unknown_jurisdiction_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_absence_records("TX")
