"""New York ingestion: legislation.nysenate.gov Open Legislation API.

The third integration shape in the corpus, and the reason New York was kept
despite supporting the fewest scenarios (DL-6). Federal is structured XML with a
separate versioning feed; California and Ohio are server-rendered HTML that must
be parsed; New York is an authenticated JSON API returning a ragged tree.

Requires a free API key in `NY_SENATE_API_KEY`. The key is never written into a
fixture; the response body does not contain it, and that is asserted when
fixtures are captured.

Two things this API does better than the others:

  activeDate  a published effective date, so nothing has to be inferred from a
              commencement rule
  parents     the ancestor chain, so section_path is read rather than assembled

And one it does worse: **`text` contains literal backslash-n sequences rather
than newlines.** Left alone they survive into every chunk and into any answer
that quotes the statute.

Paid Family Leave lives in the Workers' Compensation Law, Article 9, which is
also why a question about filing a comp claim retrieves plausible-looking
statute (`out-of-scope-006`).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import date
from pathlib import Path

from ingest.models import SourceDocument

API_ROOT = "https://legislation.nysenate.gov/api/3/laws"
USER_AGENT = "controlling-authority/0.1 (portfolio project; contact via GitHub)"
CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "ny"

# Citation strings the scenario ground truth was written against.
NY_LAW_NAMES = {"WKC": "N.Y. Workers' Comp. Law"}


def _clean(text: str) -> str:
    # The API escapes newlines as two characters. Decode before collapsing, or
    # the literal sequences are preserved as text.
    text = text.replace("\\n", " ").replace("\\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_ny_section(payload: dict, observed_on: date) -> SourceDocument:
    if not payload.get("success"):
        raise ValueError("unsuccessful API payload; the location may not exist")

    result = payload.get("result") or {}
    law_id = result.get("lawId")
    location = result.get("locationId")
    raw_text = result.get("text") or ""
    if not raw_text.strip():
        raise ValueError(f"{law_id} {location}: no text in payload")

    if law_id not in NY_LAW_NAMES:
        raise ValueError(f"unmapped New York law {law_id!r}")

    text = _clean(raw_text)
    # Strip the section heading the body repeats: "§ 204. Disability and family
    # leave during employment."
    text = re.sub(rf"^§\s*{re.escape(str(location))}\.\s*[^.]*\.\s*", "", text).strip()

    section_path = [
        p.get("title", "").strip()
        for p in (result.get("parents") or [])
        if p.get("title")
    ]

    active = result.get("activeDate")
    if not active:
        raise ValueError(f"{law_id} {location}: no activeDate; refusing to infer one")

    citation = f"{NY_LAW_NAMES[law_id]} {location}"
    return SourceDocument(
        doc_id=f"ny:{law_id.lower()}-{location}",
        citation=citation,
        authority_layer="state",
        jurisdiction="NY",
        section_path=section_path or [NY_LAW_NAMES[law_id]],
        heading=citation,
        text=text,
        content_status="substantive",
        effective_from=date.fromisoformat(active),
        effective_from_is_floor=False,
        observed_on=observed_on,
        source_url=f"https://www.nysenate.gov/legislation/laws/{law_id}/{location}",
        source_note=f"{result.get('title', '')} (activeDate {active})".strip(),
    )


def fetch_section(
    law_id: str, location: str, observed_on: date | None = None
) -> SourceDocument:
    """Fetch one section. Network, cached on disk."""
    observed_on = observed_on or date.today()
    key = os.environ.get("NY_SENATE_API_KEY")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{law_id.lower()}_{location}.json"

    if cached.exists():
        payload = json.loads(cached.read_text())
    else:
        if not key:
            raise RuntimeError("NY_SENATE_API_KEY is not set")
        request = urllib.request.Request(
            f"{API_ROOT}/{law_id}/{location}?key={key}",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
        # The key must never reach disk with the cached body.
        if key in json.dumps(payload):
            raise RuntimeError("API key present in response body; refusing to cache")
        cached.write_text(json.dumps(payload, indent=1))

    return parse_ny_section(payload, observed_on=observed_on)
