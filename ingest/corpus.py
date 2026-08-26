"""Assemble every source into one corpus.

Four adapters, four shapes, one contract. Everything downstream — chunking,
embedding, retrieval — sees only `SourceDocument` and knows nothing about where
a document came from.

Federal ingestion needs the network on a cold cache, so it is behind a flag.
Everything else reads from disk.
"""

from __future__ import annotations

from datetime import date

from ingest.absence import load_absence_records
from ingest.company_handbook import load_handbook
from ingest.federal_ecfr import fetch_part
from ingest.models import SourceDocument
from ingest.state_ca import fetch_section as fetch_ca
from ingest.state_ny import fetch_section as fetch_ny

# The federal snapshot the corpus is built from. Pinned rather than defaulting
# to today: the eCFR cache key embeds the snapshot date, so a floating default
# means every run fetches a date nobody has cached, and two runs on different
# days silently compare different text.
FEDERAL_SNAPSHOT = date(2026, 8, 1)

# Scoped by what the scenario set cites, per the plan's stopping rule.
# Scope is read off the scenario set. Three sections were added in Phase 5.5
# after verification found claims that no ingested text supported:
#   LAB 246     paid sick leave, which conflict-019 and conflict-020 rest on
#   GOV 12945.7 bereavement, which conflict-002 rests on
#   WKC 203     the eligibility rule conflict-006 and ambiguous-015 cite 204 for
CA_SECTIONS = [
    ("GOV", "12945.2"), ("GOV", "12945"), ("GOV", "12945.7"),
    ("LAB", "227.3"), ("LAB", "246"), ("ELEC", "14000"),
]
# 203 sets eligibility; 204 sets the benefit 203 points to. Both are needed:
# scenarios about "am I eligible" cite 203, scenarios about "how much" cite 204.
NY_SECTIONS = [("WKC", "203"), ("WKC", "204")]
FEDERAL_PART = 825
ABSENCE_JURISDICTIONS = ["OH"]


def build_corpus(
    observed_on: date | None = None, include_federal: bool = True
) -> list[SourceDocument]:
    """Every document in the corpus.

    `include_federal=False` skips the eCFR pull, which is the only source large
    enough for a cold fetch to be slow.
    """
    observed_on = observed_on or date.today()
    docs: list[SourceDocument] = []

    if include_federal:
        docs += fetch_part(FEDERAL_PART, as_of=FEDERAL_SNAPSHOT)

    for code, section in CA_SECTIONS:
        docs += [fetch_ca(code, section, observed_on=observed_on)]
    for law_id, location in NY_SECTIONS:
        docs += [fetch_ny(law_id, location, observed_on=observed_on)]
    for jurisdiction in ABSENCE_JURISDICTIONS:
        docs += load_absence_records(jurisdiction, observed_on=observed_on)

    docs += load_handbook(observed_on=observed_on)

    ids = [d.doc_id for d in docs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate doc_ids across sources: {sorted(duplicates)}")

    return docs


def corpus_summary(docs: list[SourceDocument]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for d in docs:
        key = f"{d.authority_layer}:{d.jurisdiction}"
        summary[key] = summary.get(key, 0) + 1
    summary["total"] = len(docs)
    summary["reserved"] = sum(1 for d in docs if d.content_status == "reserved")
    summary["absent"] = sum(1 for d in docs if d.content_status == "absent")
    return summary
