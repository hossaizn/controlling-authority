"""Federal ingestion tests.

Everything here runs against committed fixtures. Ingestion must never touch the
network in a test run: a suite that depends on a government API is a suite that
fails for reasons unrelated to the code, and one that quietly passes when the
API changes shape underneath it.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ingest.federal_ecfr import (
    effective_dates_from_versions,
    end_dates_from_versions,
    parse_ecfr_part,
)

FIXTURES = Path(__file__).parent / "fixtures"
PART_XML = FIXTURES / "ecfr_title29_part825_2026-08-01.xml"
VERSIONS_JSON = FIXTURES / "ecfr_versions_title29_part825.json"
SNAPSHOT = date(2026, 8, 1)


@pytest.fixture(scope="module")
def docs():
    versions = json.loads(VERSIONS_JSON.read_text())
    dates = effective_dates_from_versions(versions, as_of=SNAPSHOT)
    return parse_ecfr_part(PART_XML.read_bytes(), snapshot_date=SNAPSHOT, effective_dates=dates)


def test_every_section_is_emitted(docs) -> None:
    """79 sections in the fixture, including the reserved ones."""
    assert len(docs) == 79


def test_citation_comes_from_the_source_not_from_us(docs) -> None:
    """Ground truth is written against these strings. Reconstructing the format
    ourselves would fail correct answers on a formatting difference."""
    d = next(x for x in docs if x.source_id == "29 CFR 825.200")
    assert d.citation == "29 CFR 825.200"
    assert d.heading.startswith("§ 825.200")


def test_section_path_preserves_hierarchy(docs) -> None:
    d = next(x for x in docs if x.source_id == "29 CFR 825.200")
    assert d.section_path[0] == "Part 825"
    assert d.section_path[1].startswith("Subpart B")
    assert d.section_path[-1].startswith("§ 825.200")


def test_layer_and_jurisdiction_are_federal(docs) -> None:
    assert {d.authority_layer for d in docs} == {"federal"}
    assert {d.jurisdiction for d in docs} == {"US"}


def test_reserved_sections_are_marked_not_dropped(docs) -> None:
    """A gap in the numbering should be a visible fact about the corpus, not
    something that looks like a parser failure."""
    reserved = {d.source_id for d in docs if d.is_reserved}
    assert reserved == {
        "29 CFR 825.103",
        "29 CFR 825.116-825.118",
        "29 CFR 825.208",
    }
    assert all(not d.text.strip() for d in docs if d.is_reserved)


def test_substantive_sections_carry_their_text(docs) -> None:
    d = next(x for x in docs if x.source_id == "29 CFR 825.108")
    assert "public agency" in d.text.lower()
    # The heading must not be duplicated into the body, or every chunk starts
    # with a near-identical string and retrieval degrades.
    assert not d.text.startswith("§ 825.108")


def test_effective_from_uses_real_amendment_dates(docs) -> None:
    """The whole point of using the versions endpoint.

    A snapshot alone only tells you text was in force on the day you looked.
    """
    d = next(x for x in docs if x.source_id == "29 CFR 825.100")
    assert d.effective_from == date(2016, 12, 1)
    assert d.observed_on == SNAPSHOT
    # Distinct concepts; conflating them is the bug this test exists to catch.
    assert d.effective_from != d.observed_on


def test_current_text_has_no_end_date(docs) -> None:
    d = next(x for x in docs if x.source_id == "29 CFR 825.100")
    assert d.effective_to is None
    assert d.in_force_on(SNAPSHOT)


def test_effective_dates_are_the_latest_amendment_at_or_before_the_snapshot() -> None:
    versions = json.loads(VERSIONS_JSON.read_text())
    early = effective_dates_from_versions(versions, as_of=date(2018, 1, 1))
    late = effective_dates_from_versions(versions, as_of=date(2026, 8, 1))
    # Amendments after the as_of date must not leak backwards into it.
    assert all(v <= date(2018, 1, 1) for v in early.values())
    assert all(v <= date(2026, 8, 1) for v in late.values())
    changed = [k for k in early if k in late and early[k] != late[k]]
    assert changed, "no section changed between 2018 and 2026; fixture may be wrong"


def test_a_snapshot_before_a_later_amendment_gets_an_end_date() -> None:
    """Point-in-time ingestion must bound the text it read.

    Reading a 2018 snapshot and leaving effective_to as None would assert that
    2018 text is still in force today.
    """
    versions = json.loads(VERSIONS_JSON.read_text())
    as_of = date(2018, 1, 1)
    old = parse_ecfr_part(
        PART_XML.read_bytes(),
        snapshot_date=as_of,
        effective_dates=effective_dates_from_versions(versions, as_of=as_of),
        end_dates=end_dates_from_versions(versions, as_of=as_of),
    )
    superseded = [d for d in old if d.effective_to is not None]
    assert superseded, "expected some sections to have been amended after 2018"
    for d in superseded:
        assert d.effective_to >= d.effective_from
        assert not d.in_force_on(date(2026, 8, 1))


def test_sections_missing_from_the_versions_feed_are_not_silently_dated(docs) -> None:
    """A section with no amendment record must not be given a made-up date."""
    versions = json.loads(VERSIONS_JSON.read_text())
    dates = effective_dates_from_versions(versions, as_of=SNAPSHOT)
    with pytest.raises(ValueError, match="no amendment date"):
        parse_ecfr_part(
            PART_XML.read_bytes(),
            snapshot_date=SNAPSHOT,
            effective_dates={k: v for k, v in list(dates.items())[:5]},
        )
