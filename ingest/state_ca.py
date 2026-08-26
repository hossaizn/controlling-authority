"""California ingestion: leginfo.legislature.ca.gov.

Server-rendered HTML, no JavaScript and no API. Verified 2026-08-25: the statute
body is present in the HTTP response.

Structure, confirmed against the live pages rather than assumed. Inside
`#codeLawSectionNoHead`:

    h4  code name, then the TITLE / DIVISION / PART / CHAPTER hierarchy
    h5  ARTICLE
    h6  the section number
    the final <div> holds the section text, ending in a credit line

**Effective dates are published, sometimes.** The credit line ends with
"Effective January 1, 2023" where the legislature stated one. Where it does not,
only the chapter year is given and the date has to be inferred from California's
ordinary commencement rule, 1 January following enactment.

That inference is marked, because it is not always right: Gov Code 12945 took
effect on 30 June 2022, and assuming the ordinary rule would have dated it six
months late. Only published dates are treated as authoritative.

Ingestion is scoped to the sections the scenario set cites. The plan's stopping
rule is that ingestion ends when the eval is answerable, not when the corpus
feels complete.
"""

from __future__ import annotations

import re
import time
import urllib.request
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

from ingest.models import SourceDocument

BASE = "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml"
USER_AGENT = "controlling-authority/0.1 (portfolio project; contact via GitHub)"
CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "ca"

# Citation strings the scenario ground truth was written against. These must
# match exactly; reformatting here fails correct answers on cosmetics.
CA_CODE_NAMES = {
    "GOV": "Cal. Gov. Code",
    "LAB": "Cal. Lab. Code",
    "ELEC": "Cal. Elec. Code",
}

# "(Amended by Stats. 2022, Ch. 748, Sec. 1. (AB 1041) Effective January 1, 2023.)"
CREDIT_LINE = re.compile(r"\((?:Amended|Added|Enacted|Repealed)[^()]*(?:\([^()]*\)[^()]*)*\)\s*$")
EFFECTIVE_DATE = re.compile(r"Effective\s+([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})")
CHAPTER_YEAR = re.compile(r"Stats\.\s*(\d{4})")

MONTHS = {
    m: i
    for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"],
        start=1,
    )
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _effective_from(credit: str) -> tuple[date, bool]:
    """(date, is_floor). Published dates win; otherwise infer and say so."""
    published = EFFECTIVE_DATE.search(credit)
    if published:
        month, day, year = published.groups()
        return date(int(year), MONTHS[month], int(day)), False

    chaptered = CHAPTER_YEAR.search(credit)
    if not chaptered:
        raise ValueError(f"no chapter year in credit line: {credit[:120]!r}")
    # California's ordinary rule. Wrong for urgency statutes, hence the flag.
    return date(int(chaptered.group(1)) + 1, 1, 1), True


def parse_ca_section(
    html: str, code: str, section: str, observed_on: date
) -> SourceDocument:
    if code not in CA_CODE_NAMES:
        raise ValueError(f"unmapped California code {code!r}")

    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("#codeLawSectionNoHead")
    blocks = container.find_all("div", recursive=False) if container else []
    body_text = _clean(blocks[-1].get_text(" ")) if blocks else ""

    # leginfo answers 200 with a shell page for an unknown section. An empty
    # document would enter the corpus and retrieve as a plausible near-miss.
    if not body_text.startswith(f"{section}."):
        raise ValueError(
            f"no statute text for {code} {section}: page did not contain the section body"
        )

    # Hierarchy: the h4s after the code name, plus any h5.
    headers = [_clean(h.get_text(" ")) for h in container.select("h4")]
    section_path = [h for h in headers[1:] if h]
    section_path += [_clean(h.get_text(" ")) for h in container.select("h5")]

    credit_match = CREDIT_LINE.search(body_text)
    credit = credit_match.group(0) if credit_match else ""
    if not credit:
        raise ValueError(f"{code} {section}: no credit line, cannot date the section")

    text = body_text[: credit_match.start()].strip()
    text = text[len(section) + 1 :].strip()  # drop the leading "12945.2."

    effective_from, is_floor = _effective_from(credit)
    citation = f"{CA_CODE_NAMES[code]} {section}"

    return SourceDocument(
        doc_id=f"ca:{code.lower()}-{section}",
        citation=citation,
        authority_layer="state",
        jurisdiction="CA",
        section_path=section_path or [CA_CODE_NAMES[code]],
        heading=citation,
        text=text,
        content_status="substantive",
        effective_from=effective_from,
        effective_from_is_floor=is_floor,
        observed_on=observed_on,
        source_url=f"{BASE}?lawCode={code}&sectionNum={section}",
        source_note=credit,
    )


def fetch_section(code: str, section: str, observed_on: date | None = None) -> SourceDocument:
    """Fetch one section. Network, cached on disk."""
    observed_on = observed_on or date.today()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{code.lower()}_{section}.html"
    if cached.exists():
        html = cached.read_text(errors="ignore")
    else:
        request = urllib.request.Request(
            f"{BASE}?lawCode={code}&sectionNum={section}",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            html = response.read().decode("utf-8", errors="ignore")
        cached.write_text(html)
        time.sleep(1.5)  # this is a scrape of a public site, not an API
    return parse_ca_section(html, code, section, observed_on)
