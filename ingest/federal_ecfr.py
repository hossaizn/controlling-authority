"""Federal ingestion: 29 CFR Part 825 (FMLA) from the eCFR API.

Two endpoints are used, and the distinction between them is the reason this
adapter is more than an XML parse:

  /versioner/v1/full/{date}/title-29.xml?part=825
      The text as it stood on a given day.

  /versioner/v1/versions/title-29.json?part=825
      Per-section amendment history.

A snapshot alone answers "what did this say on 1 August 2026". It cannot answer
"since when", because the text it returns may have been unchanged since 2016.
Effective dating therefore comes from the versions feed, and the snapshot date is
recorded separately as `observed_on`. Conflating the two would make every federal
provision look as though it began on the day we happened to fetch it.

Verified against the live API on 2026-08-25: the part contains 79 sections in 8
subparts, every section carries a `hierarchy_metadata` citation, and the versions
feed reports 16 distinct amendment dates between 2016-12-01 and 2025-01-15.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

from lxml import etree

from ingest.http import fetch as http_fetch
from ingest.models import SourceDocument

API_ROOT = "https://www.ecfr.gov/api/versioner/v1"

# Cached raw pulls. Gitignored: regenerable, and large.
CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "ecfr"

# eCFR structure, confirmed against the live document rather than assumed:
#   DIV5 = PART, DIV6 = SUBPART, DIV8 = SECTION
PART, SUBPART, SECTION = "DIV5", "DIV6", "DIV8"

# Elements that are not operative text. HEAD is the heading, held separately.
# CITA, SOURCE and AUTH are provenance, captured into source_note rather than
# discarded: they carry the Federal Register history a reader needs to check a
# citation.
NON_BODY = {"HEAD", "CITA", "SOURCE", "AUTH"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _head(el) -> str:
    return _clean(el.findtext("HEAD") or "")


def baseline_load_date(versions_payload: dict) -> date | None:
    """The date the whole part entered the electronic record, if there is one.

    The eCFR versions feed stamps every section of a part with a shared date on
    initial load. For Part 825 that is 2016-12-01, carried by all 79 sections,
    while the regulation itself dates to 2013. Treating it as an amendment date
    would report when the text entered the database rather than when it began.

    Detected structurally rather than hardcoded: the date every section shares.
    """
    per_section: dict[str, set[date]] = {}
    for row in versions_payload.get("content_versions", []):
        identifier, raw = row.get("identifier"), row.get("amendment_date")
        if identifier and raw:
            per_section.setdefault(identifier, set()).add(date.fromisoformat(raw))
    if len(per_section) < 2:
        return None
    shared = set.intersection(*per_section.values())
    return min(shared) if shared else None


def effective_dates_from_versions(
    versions_payload: dict, as_of: date
) -> dict[str, date]:
    """Latest amendment date at or before `as_of`, per section identifier.

    The feed carries several rows per section, one per issue, often sharing an
    amendment date. Only amendments at or before the snapshot may apply: a later
    one describes text we did not fetch.
    """
    # The feed marks 36 of 132 rows `substantive: false` and none `removed`.
    # Both flags are deliberately ignored, and the reason differs.
    #
    # Non-substantive amendments (nomenclature changes, technical corrections)
    # still alter the text, and this field answers "when did this wording take
    # effect", not "when did the rule change". Filtering them would date a
    # provision to before its current wording existed.
    #
    # `removed` never occurs in this part. A removed section would need to end
    # rather than begin on that date, which this function cannot express. If a
    # part with removals is ever ingested, that must be handled before trusting
    # any point-in-time answer touching it.
    latest: dict[str, date] = {}
    for row in versions_payload.get("content_versions", []):
        identifier = row.get("identifier")
        raw = row.get("amendment_date")
        if not identifier or not raw:
            continue
        amended = date.fromisoformat(raw)
        if amended > as_of:
            continue
        if identifier not in latest or amended > latest[identifier]:
            latest[identifier] = amended
    return latest


def end_dates_from_versions(
    versions_payload: dict, as_of: date
) -> dict[str, date]:
    """When the text in force at `as_of` stopped being in force.

    The earliest amendment strictly after the snapshot, minus a day. Sections
    with no later amendment are absent from the result, meaning still current.

    Without this, ingesting a 2018 snapshot would assert that 2018 text governs
    today, which is precisely the failure the superseded scenario slice tests.
    """
    ends: dict[str, date] = {}
    for row in versions_payload.get("content_versions", []):
        identifier = row.get("identifier")
        raw = row.get("amendment_date")
        if not identifier or not raw:
            continue
        amended = date.fromisoformat(raw)
        if amended <= as_of:
            continue
        if identifier not in ends or amended < ends[identifier]:
            ends[identifier] = amended
    return {k: v - timedelta(days=1) for k, v in ends.items()}


def parse_ecfr_part(
    xml_bytes: bytes,
    snapshot_date: date,
    effective_dates: dict[str, date],
    end_dates: dict[str, date] | None = None,
    baseline_date: date | None = None,
) -> list[SourceDocument]:
    """Parse one part into a document per section.

    Raises rather than guessing when a section has no amendment record. A
    fabricated effective date would silently corrupt every point-in-time answer
    that touched it, and would be invisible downstream.
    """
    end_dates = end_dates or {}
    floor_date = baseline_date
    root = etree.fromstring(xml_bytes)
    if root.tag != PART:
        raise ValueError(f"expected a {PART} (PART) root, got {root.tag!r}")

    part_label = f"Part {root.get('N')}"
    docs: list[SourceDocument] = []

    for subpart in root.findall(f".//{SUBPART}"):
        subpart_label = _head(subpart)
        for section in subpart.findall(SECTION):
            identifier = section.get("N")
            heading = _head(section)

            meta = json.loads(section.get("hierarchy_metadata") or "{}")
            citation = meta.get("citation")
            if not citation:
                raise ValueError(f"{identifier}: no citation in hierarchy_metadata")

            # Body text is every child that is not the heading or provenance.
            # The first version read only <P>, which silently dropped anything
            # held in another element, including tables.
            blocks = [
                _clean("".join(child.itertext()))
                for child in section
                if child.tag not in NON_BODY
            ]
            text = "\n\n".join(b for b in blocks if b)

            # Provenance the source prints about itself, kept for citation.
            notes = [
                _clean("".join(n.itertext()))
                for n in section
                if n.tag in {"CITA", "SOURCE"}
            ]
            source_note = " ".join(n for n in notes if n)

            is_reserved = "[Reserved]" in heading
            if is_reserved:
                text = ""

            if identifier not in effective_dates:
                raise ValueError(
                    f"{identifier}: no amendment date in the versions feed; refusing "
                    "to invent one"
                )

            effective_from = effective_dates[identifier]
            docs.append(
                SourceDocument(
                    doc_id=f"us:29-cfr-{identifier}",
                    citation=citation,
                    authority_layer="federal",
                    jurisdiction="US",
                    # Ancestors only: the heading is held separately and must
                    # not be repeated here.
                    section_path=[part_label, subpart_label],
                    heading=heading,
                    text=text,
                    content_status="reserved" if is_reserved else "substantive",
                    effective_from=effective_from,
                    effective_to=end_dates.get(identifier),
                    effective_from_is_floor=(
                        floor_date is not None and effective_from == floor_date
                    ),
                    observed_on=snapshot_date,
                    # Point-in-time documents must link to the text as it stood,
                    # not to /current/, which would show today's wording for a
                    # provision that ended years ago.
                    source_url=(
                        f"https://www.ecfr.gov/on/{snapshot_date.isoformat()}"
                        f"/title-29/section-{identifier}"
                    ),
                    source_note=source_note,
                )
            )

    return docs


def _get(url: str, cache_name: str) -> bytes:
    """Fetch with an on-disk cache. The same snapshot is never fetched twice."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / cache_name
    if cached.exists():
        return cached.read_bytes()
    payload = http_fetch(url, timeout=120)
    cached.write_bytes(payload)
    return payload


def fetch_part(part: int = 825, as_of: date | None = None) -> list[SourceDocument]:
    """Fetch and parse a part as it stood on a given date. Network, cached."""
    as_of = as_of or date.today()
    xml = _get(
        f"{API_ROOT}/full/{as_of.isoformat()}/title-29.xml?part={part}",
        f"title29_part{part}_{as_of.isoformat()}.xml",
    )
    versions = json.loads(
        _get(
            f"{API_ROOT}/versions/title-29.json?part={part}",
            f"title29_part{part}_versions.json",
        )
    )
    return parse_ecfr_part(
        xml,
        snapshot_date=as_of,
        effective_dates=effective_dates_from_versions(versions, as_of),
        end_dates=end_dates_from_versions(versions, as_of),
        baseline_date=baseline_load_date(versions),
    )
