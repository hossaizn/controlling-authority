"""Absence records: a layer positively stating that it says nothing.

The problem this solves. Ohio has no state family leave statute for private
employers. Ask the corpus about Ohio family leave and, without these records,
retrieval returns nothing. But "nothing came back" is what a retrieval failure
also looks like, and the two demand opposite responses: one is "I could not find
this", the other is "there is nothing to find, and that is the answer".

So an absence is a document. It carries text, it is retrievable, and it says
what the absence means for precedence. `content_status` is "absent" rather than
"substantive" so it can never be mistaken for a statute, and never cited as one.

These records are hand-written and, like every other statutory claim in this
project, unverified until checked (DL-3). `verified_on` stays null until then.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml

from ingest.models import SourceDocument

ABSENCE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "absence"

# What the absence means once precedence is applied.
AbsenceEffect = Literal["federal_controls", "employer_policy_controls"]

# An absence is a standing fact about a body of law, not a dated provision. It
# is treated as in force for any query the corpus can answer, so a point-in-time
# question about 2023 still learns that Ohio had no such statute.
ABSENCE_EFFECTIVE_FROM = date(1900, 1, 1)


def load_absence_records(
    jurisdiction: str, observed_on: date | None = None
) -> list[SourceDocument]:
    observed_on = observed_on or date.today()
    path = ABSENCE_DIR / f"{jurisdiction.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no absence records for {jurisdiction}")

    entries = yaml.safe_load(path.read_text()) or []
    docs: list[SourceDocument] = []
    for entry in entries:
        topic = entry["topic"]
        effect = entry["effect"]
        if effect not in ("federal_controls", "employer_policy_controls"):
            raise ValueError(f"{jurisdiction} {topic}: unknown effect {effect!r}")

        text = " ".join(entry["text"].split())
        if not text:
            raise ValueError(f"{jurisdiction} {topic}: absence records must carry text")

        docs.append(
            SourceDocument(
                doc_id=f"{jurisdiction.lower()}:absence-{topic}",
                # Deliberately not a statutory citation: there is no statute to
                # cite. An answer quoting this should read as a statement about
                # the law's silence, not as authority.
                citation=f"{jurisdiction.upper()} (no state provision: {topic})",
                authority_layer="state",
                jurisdiction=jurisdiction.upper(),
                section_path=[f"{jurisdiction.upper()} state law", "Recorded absences"],
                heading=f"No {jurisdiction.upper()} state provision: {topic}",
                text=text,
                content_status="absent",
                effective_from=ABSENCE_EFFECTIVE_FROM,
                effective_from_is_floor=True,
                observed_on=observed_on,
                source_url="",
                source_note=f"effect={effect}; verified_on={entry.get('verified_on')}",
            )
        )
    return docs
