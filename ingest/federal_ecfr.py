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
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from lxml import etree

from ingest.models import SourceDocument

API_ROOT = "https://www.ecfr.gov/api/versioner/v1"
USER_AGENT = "controlling-authority/0.1 (portfolio project; contact via GitHub)"

# Cached raw pulls. Gitignored: regenerable, and large.
CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "ecfr"

# eCFR structure, confirmed against the live document rather than assumed:
#   DIV5 = PART, DIV6 = SUBPART, DIV8 = SECTION
PART, SUBPART, SECTION = "DIV5", "DIV6", "DIV8"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _head(el) -> str:
    return _clean(el.findtext("HEAD") or "")


def effective_dates_from_versions(
    versions_payload: dict, as_of: date
) -> dict[str, date]:
    """Latest amendment date at or before `as_of`, per section identifier.

    The feed carries several rows per section, one per issue, often sharing an
    amendment date. Only amendments at or before the snapshot may apply: a later
    one describes text we did not fetch.
    """
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
) -> list[SourceDocument]:
    """Parse one part into a document per section.

    Raises rather than guessing when a section has no amendment record. A
    fabricated effective date would silently corrupt every point-in-time answer
    that touched it, and would be invisible downstream.
    """
    end_dates = end_dates or {}
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

            # Body text excludes HEAD, so the heading is not duplicated into
            # every chunk. Paragraph boundaries are preserved for the chunker.
            paragraphs = [
                _clean("".join(p.itertext()))
                for p in section.iterdescendants("P")
            ]
            text = "\n\n".join(p for p in paragraphs if p)

            is_reserved = "[Reserved]" in heading
            if is_reserved:
                text = ""

            if identifier not in effective_dates:
                raise ValueError(
                    f"{identifier}: no amendment date in the versions feed; refusing "
                    "to invent one"
                )

            docs.append(
                SourceDocument(
                    source_id=citation,
                    citation=citation,
                    authority_layer="federal",
                    jurisdiction="US",
                    section_path=[part_label, subpart_label, heading],
                    heading=heading,
                    text=text,
                    effective_from=effective_dates[identifier],
                    effective_to=end_dates.get(identifier),
                    observed_on=snapshot_date,
                    source_url=f"https://www.ecfr.gov/current/title-29/section-{identifier}",
                    is_reserved=is_reserved,
                )
            )

    return docs


def _get(url: str, cache_name: str) -> bytes:
    """Fetch with an on-disk cache. The same snapshot is never fetched twice."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / cache_name
    if cached.exists():
        return cached.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
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
    )
