"""Federal ingestion tests.

Everything here runs against committed fixtures; conftest blocks the network.

The date assertions pin actual values. The first version of this file asserted
only relationships (`effective_to >= effective_from`, "some are non-null"), and
review demonstrated the hole by mutating the day-offset arithmetic in two
directions and removing it entirely: the suite passed every time. Relationship
assertions on derived dates prove nothing about the derivation.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from ingest.federal_ecfr import (
    baseline_load_date,
    effective_dates_from_versions,
    end_dates_from_versions,
    parse_ecfr_part,
)

FIXTURES = Path(__file__).parent / "fixtures"
PART_XML = FIXTURES / "ecfr_title29_part825_2026-08-01.xml"
VERSIONS_JSON = FIXTURES / "ecfr_versions_title29_part825.json"
SNAPSHOT = date(2026, 8, 1)


@pytest.fixture(scope="module")
def versions() -> dict:
    return json.loads(VERSIONS_JSON.read_text())


def _parse(versions: dict, as_of: date):
    return parse_ecfr_part(
        PART_XML.read_bytes(),
        snapshot_date=as_of,
        effective_dates=effective_dates_from_versions(versions, as_of=as_of),
        end_dates=end_dates_from_versions(versions, as_of=as_of),
        baseline_date=baseline_load_date(versions),
    )


@pytest.fixture(scope="module")
def docs(versions):
    return _parse(versions, SNAPSHOT)


def by_id(docs, ident: str):
    return next(d for d in docs if d.doc_id == f"us:29-cfr-{ident}")


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_every_section_is_emitted(docs) -> None:
    assert len(docs) == 79


def test_citation_comes_from_the_source_not_from_us(docs) -> None:
    d = by_id(docs, "825.200")
    assert d.citation == "29 CFR 825.200"
    assert d.heading.startswith("§ 825.200")


def test_section_path_holds_ancestors_only(docs) -> None:
    """Repeating the heading in the path duplicates it in every chunk."""
    d = by_id(docs, "825.200")
    assert d.section_path == ["Part 825", d.section_path[1]]
    assert d.section_path[1].startswith("Subpart B")
    assert d.heading not in d.section_path


def test_layer_and_jurisdiction_are_federal(docs) -> None:
    assert {d.authority_layer for d in docs} == {"federal"}
    assert {d.jurisdiction for d in docs} == {"US"}


def test_reserved_sections_are_marked_not_dropped(docs) -> None:
    reserved = {d.citation for d in docs if d.content_status == "reserved"}
    assert reserved == {"29 CFR 825.103", "29 CFR 825.116-825.118", "29 CFR 825.208"}
    assert all(not d.text.strip() for d in docs if d.content_status == "reserved")


def test_substantive_sections_carry_their_text(docs) -> None:
    d = by_id(docs, "825.108")
    assert "public agency" in d.text.lower()
    assert not d.text.startswith("§ 825.108")


def test_provenance_notes_are_captured(docs) -> None:
    """CITA and SOURCE elements were dropped in the first version, losing the
    Federal Register history a reader needs to check a citation."""
    with_notes = [d for d in docs if d.source_note]
    assert with_notes, "no section captured provenance"
    assert any("FR" in d.source_note for d in with_notes)


def test_content_hash_is_populated(docs) -> None:
    substantive = [d for d in docs if d.content_status == "substantive"]
    assert all(d.content_hash for d in substantive)
    assert len({d.content_hash for d in substantive}) > 70, "hashes should be distinct"


def test_point_in_time_url_is_dated_not_current(versions) -> None:
    """A document that ceased in 2018 must not link to today's wording."""
    old = _parse(versions, date(2018, 1, 1))
    assert all("/on/2018-01-01/" in d.source_url for d in old)
    assert all("/current/" not in d.source_url for d in old)


# --------------------------------------------------------------------------
# Dates: pinned values, not relationships
# --------------------------------------------------------------------------


def test_baseline_load_date_is_detected(versions) -> None:
    """Every section of the part shares 2016-12-01 on initial load."""
    assert baseline_load_date(versions) == date(2016, 12, 1)


def test_baseline_dated_sections_are_flagged_as_floors(docs) -> None:
    """These report when the text entered the electronic record, not when the
    provision began. Presenting them as amendment dates repeats the DL-8 error
    one level down."""
    d = by_id(docs, "825.200")
    assert d.effective_from == date(2016, 12, 1)
    assert d.effective_from_is_floor is True
    floors = [x for x in docs if x.effective_from_is_floor]
    assert len(floors) >= 70, "most of this part sits on the baseline load"


def test_genuinely_amended_sections_are_not_flagged_as_floors(docs) -> None:
    amended = [d for d in docs if not d.effective_from_is_floor]
    assert amended, "some sections have real amendment dates"
    assert all(d.effective_from > date(2016, 12, 1) for d in amended)


def test_effective_from_is_the_exact_latest_amendment_at_or_before_snapshot(
    versions,
) -> None:
    early = _parse(versions, date(2018, 1, 1))
    # Amended 2017-01-18, and again on 2018-01-02. At a 2018-01-01 snapshot the
    # earlier amendment is the one in force.
    assert by_id(early, "825.300").effective_from == date(2017, 1, 18)
    assert by_id(early, "825.104").effective_from == date(2017, 1, 9)


def test_effective_to_is_the_exact_day_before_the_next_amendment(versions) -> None:
    """Pins the offset. Mutating it in either direction must fail here."""
    early = _parse(versions, date(2018, 1, 1))
    # Next amendment 2018-06-27, so the text in force ends the day before.
    assert by_id(early, "825.120").effective_to == date(2018, 6, 26)
    # Ends on the snapshot date itself: superseded the very next day.
    assert by_id(early, "825.300").effective_to == date(2018, 1, 1)
    # Exactly two sections were superseded after a 2018-01-01 snapshot.
    assert sum(1 for d in early if d.effective_to is not None) == 2


def test_in_force_on_is_inclusive_at_both_edges(versions) -> None:
    early = _parse(versions, date(2018, 1, 1))
    d = by_id(early, "825.120")
    start, end = d.effective_from, d.effective_to
    assert d.in_force_on(start)
    assert d.in_force_on(end)
    assert not d.in_force_on(start - timedelta(days=1))
    assert not d.in_force_on(end + timedelta(days=1))
    assert end == date(2018, 6, 26) and not d.in_force_on(date(2018, 6, 27))


def test_a_snapshot_on_an_amendment_date_includes_that_amendment(versions) -> None:
    """The inclusive boundary of `amendment_date <= as_of`.

    Added after mutation testing: flipping that comparison to strict survived
    the suite, because no snapshot fell exactly on an amendment date. Text
    amended on the day you ask about is in force that day.
    """
    on_amendment = date(2018, 6, 27)
    docs = _parse(versions, on_amendment)
    d = by_id(docs, "825.120")
    assert d.effective_from == on_amendment
    assert d.in_force_on(on_amendment)
    # And it must not be treated as a future amendment that ends the text.
    assert d.effective_to is None or d.effective_to > on_amendment


def test_current_text_has_no_end_date(docs) -> None:
    d = by_id(docs, "825.100")
    assert d.effective_to is None
    assert d.in_force_on(SNAPSHOT)
    assert d.in_force_on(date(2030, 1, 1))


def test_amendments_after_the_snapshot_do_not_leak_backwards(versions) -> None:
    early = effective_dates_from_versions(versions, as_of=date(2018, 1, 1))
    assert early
    assert all(v <= date(2018, 1, 1) for v in early.values())


def test_sections_missing_from_the_versions_feed_are_not_silently_dated(
    versions,
) -> None:
    dates = effective_dates_from_versions(versions, as_of=SNAPSHOT)
    with pytest.raises(ValueError, match="no amendment date"):
        parse_ecfr_part(
            PART_XML.read_bytes(),
            snapshot_date=SNAPSHOT,
            effective_dates={k: v for k, v in list(dates.items())[:5]},
        )


# --------------------------------------------------------------------------
# The guard itself
# --------------------------------------------------------------------------


def test_network_is_blocked_in_tests() -> None:
    """conftest must make a stray fetch impossible, not merely unlikely."""
    import urllib.request

    from tests.conftest import NetworkAccessDenied

    with pytest.raises((NetworkAccessDenied, OSError)):
        urllib.request.urlopen("https://www.ecfr.gov/api/versioner/v1/titles.json", timeout=5)
