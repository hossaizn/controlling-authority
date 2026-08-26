"""New York ingestion tests. Fixture only; conftest blocks the network."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ingest.state_ny import parse_ny_section

FIXTURE = Path(__file__).parent / "fixtures" / "state" / "ny_wkc_204.json"
OBSERVED = date(2026, 8, 26)


@pytest.fixture(scope="module")
def pfl():
    payload = json.loads(FIXTURE.read_text())
    return parse_ny_section(payload, observed_on=OBSERVED)


def test_citation_matches_the_scenario_ground_truth(pfl) -> None:
    assert pfl.citation == "N.Y. Workers' Comp. Law 204"
    assert pfl.doc_id == "ny:wkc-204"


def test_layer_and_jurisdiction(pfl) -> None:
    assert pfl.authority_layer == "state"
    assert pfl.jurisdiction == "NY"


def test_section_path_comes_from_the_parent_chain(pfl) -> None:
    """Depth is ragged across New York's tree and must not be assumed."""
    assert pfl.section_path == ["Workers' Compensation", "Disability Benefits"]
    assert pfl.heading not in pfl.section_path


def test_escaped_newlines_are_decoded(pfl) -> None:
    """The API returns literal backslash-n as two characters, not newlines.

    Left alone they survive into every chunk and into any answer quoting the
    text.
    """
    assert "\\n" not in pfl.text
    assert "family leave" in pfl.text.lower()


def test_section_heading_is_not_duplicated_into_the_body(pfl) -> None:
    assert not pfl.text.lstrip().startswith("§ 204.")
    assert pfl.heading.startswith("N.Y. Workers' Comp. Law 204")


def test_effective_date_is_published_not_inferred(pfl) -> None:
    """activeDate is the API's own field, so no commencement rule is guessed."""
    assert pfl.effective_from == date(2016, 4, 8)
    assert pfl.effective_from_is_floor is False


def test_document_is_substantive_and_hashed(pfl) -> None:
    assert pfl.content_status == "substantive"
    assert len(pfl.text) > 5000
    assert pfl.content_hash


def test_a_payload_without_text_raises() -> None:
    with pytest.raises(ValueError, match="no text"):
        parse_ny_section(
            {"success": True, "result": {"lawId": "WKC", "locationId": "204", "text": ""}},
            observed_on=OBSERVED,
        )


def test_an_unsuccessful_payload_raises() -> None:
    """The API answers 200 with success:false for an unknown location."""
    with pytest.raises(ValueError, match="unsuccessful"):
        parse_ny_section({"success": False, "result": {}}, observed_on=OBSERVED)


def test_the_section_title_is_stripped_not_just_the_number(pfl) -> None:
    """Reducing the strip to the section number alone left the title in the
    body, which no test noticed."""
    assert not pfl.text.startswith("Disability and family leave during employment")
    assert pfl.text.lstrip().startswith("1.")


def test_a_title_containing_a_period_does_not_over_strip() -> None:
    """The original pattern consumed everything to the first full stop."""
    payload = {
        "success": True,
        "result": {
            "lawId": "WKC",
            "locationId": "204",
            "title": "Benefits under art. 9 of this chapter",
            "activeDate": "2016-04-08",
            "parents": [],
            "text": "  § 204. Benefits under art. 9 of this chapter. 1. The first"
                    " subdivision must survive intact.",
        },
    }
    d = parse_ny_section(payload, observed_on=OBSERVED)
    assert d.text.startswith("1. The first subdivision must survive intact.")


def test_no_title_does_not_eat_the_first_subdivision() -> None:
    payload = {
        "success": True,
        "result": {
            "lawId": "WKC",
            "locationId": "204",
            "title": "",
            "activeDate": "2016-04-08",
            "parents": [],
            "text": "  § 204. 1. Disability benefits shall be payable.",
        },
    }
    d = parse_ny_section(payload, observed_on=OBSERVED)
    assert d.text.startswith("1. Disability benefits")


def test_heading_carries_the_section_title(pfl) -> None:
    """heading is what a chunk carries into the embedding. Repeating the
    citation leaves state documents with no descriptive context while federal
    ones have it."""
    assert pfl.heading == (
        "N.Y. Workers' Comp. Law 204 — Disability and family leave during employment"
    )
