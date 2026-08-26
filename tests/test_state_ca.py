"""California ingestion tests. Fixtures only; conftest blocks the network."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ingest.state_ca import CA_CODE_NAMES, parse_ca_section

FIXTURES = Path(__file__).parent / "fixtures" / "state"
OBSERVED = date(2026, 8, 26)


def _parse(name: str, code: str, section: str):
    return parse_ca_section(
        (FIXTURES / name).read_text(errors="ignore"),
        code=code,
        section=section,
        observed_on=OBSERVED,
    )


@pytest.fixture(scope="module")
def cfra():
    return _parse("ca_gov_12945.2.html", "GOV", "12945.2")


def test_citation_matches_the_scenario_ground_truth(cfra) -> None:
    """Scenario ground truth was written against these exact strings. A
    formatting difference here fails correct answers."""
    assert cfra.citation == "Cal. Gov. Code 12945.2"
    assert cfra.doc_id == "ca:gov-12945.2"


def test_layer_and_jurisdiction(cfra) -> None:
    assert cfra.authority_layer == "state"
    assert cfra.jurisdiction == "CA"


def test_section_path_is_the_code_hierarchy(cfra) -> None:
    assert cfra.section_path[0].startswith("TITLE 2")
    assert any(p.startswith("PART 2.8") for p in cfra.section_path)
    assert cfra.heading not in cfra.section_path


def test_text_is_the_statute_body(cfra) -> None:
    assert "unlawful employment practice" in cfra.text
    assert "1,250 hours" in cfra.text
    # The leading section number is the heading, not body text.
    assert not cfra.text.startswith("12945.2.")


def test_history_line_is_provenance_not_body(cfra) -> None:
    """The credit line is what a reader checks a citation against, but it is
    not operative text and must not be retrieved as though it were."""
    assert "Stats. 2022, Ch. 748" in cfra.source_note
    assert "Amended by Stats" not in cfra.text


def test_explicit_effective_date_is_used_when_published(cfra) -> None:
    assert cfra.effective_from == date(2023, 1, 1)
    assert cfra.effective_from_is_floor is False


def test_a_non_january_effective_date_is_not_normalised_away() -> None:
    """Gov Code 12945 took effect 30 June 2022, not the following 1 January.

    Assuming California's ordinary commencement rule would have dated this six
    months late. The published date is used wherever one exists.
    """
    d = _parse("ca_gov_12945.html", "GOV", "12945")
    assert d.effective_from == date(2022, 6, 30)
    assert d.effective_from_is_floor is False


def test_missing_effective_date_falls_back_and_is_flagged() -> None:
    """Older sections publish only the chapter year.

    The fallback is California's ordinary rule, 1 January following enactment.
    It can be wrong for urgency statutes, so it is marked as not authoritative
    rather than presented as a published date.
    """
    d = _parse("ca_lab_227.3.html", "LAB", "227.3")
    assert d.effective_from == date(1977, 1, 1)
    assert d.effective_from_is_floor is True
    assert "Stats. 1976" in d.source_note


def test_enacted_sections_are_dated_too() -> None:
    d = _parse("ca_elec_14000.html", "ELEC", "14000")
    assert d.citation == "Cal. Elec. Code 14000"
    assert d.effective_from == date(1995, 1, 1)
    assert d.effective_from_is_floor is True


def test_all_four_required_sections_parse() -> None:
    """Ingestion is scoped by what the scenarios cite, and nothing more."""
    wanted = [
        ("ca_gov_12945.2.html", "GOV", "12945.2", "Cal. Gov. Code 12945.2"),
        ("ca_gov_12945.html", "GOV", "12945", "Cal. Gov. Code 12945"),
        ("ca_lab_227.3.html", "LAB", "227.3", "Cal. Lab. Code 227.3"),
        ("ca_elec_14000.html", "ELEC", "14000", "Cal. Elec. Code 14000"),
    ]
    for name, code, section, citation in wanted:
        d = _parse(name, code, section)
        assert d.citation == citation
        assert d.content_status == "substantive"
        assert d.text.strip()
        assert d.content_hash


def test_code_name_map_covers_every_code_we_ingest() -> None:
    assert {"GOV", "LAB", "ELEC"} <= set(CA_CODE_NAMES)


def test_a_page_without_statute_text_raises() -> None:
    """leginfo returns 200 with a shell page for a bad section number. Silently
    producing an empty document would poison the corpus."""
    with pytest.raises(ValueError, match="no statute text"):
        parse_ca_section("<html><body>nothing here</body></html>", "GOV", "12945.2", OBSERVED)
