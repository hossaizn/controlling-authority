"""Company handbook ingestion: the fourth source, and the only authored one.

Markdown with YAML front matter, so it is by far the easiest to parse. It is
also the only layer that carries versions, which the entire superseded scenario
slice depends on: `LEAVE-004` exists as v1 and v2 with adjoining date ranges,
and the correct answer to a sick-leave question turns on which was in force.

**`DEFECTS.md` must never be ingested.** It records which handbook clauses are
deliberately wrong and what the correct resolution is. Putting it into the
corpus would place the answer key inside the thing being searched, and every
conflict scenario would become answerable by looking up the answer. The allowlist
below is a filename pattern rather than a denylist for that reason: a new
non-policy file added to this directory is excluded by default rather than
included by accident.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from ingest.models import SourceDocument

HANDBOOK_DIR = Path(__file__).resolve().parent.parent / "corpus" / "handbook"

# Allowlist, not a denylist. Anything that is not a numbered policy file is not
# part of the corpus, whatever it is called.
POLICY_FILENAME = re.compile(r"^LEAVE-\d{3}(?:-v\d+)?-[a-z0-9-]+\.md$", re.I)

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

# The editorial banner on a superseded version. It describes the document rather
# than stating a term of the policy, and retrieving it as policy text would
# mislead.
# Greedy, deliberately. A lazy quantifier here matched only the "> " marker and
# left the sentence behind, because nothing downstream forced it to consume
# more. The blockquote runs to the end of its line and any continuation lines.
SUPERSEDED_BANNER = re.compile(r"^>[ \t]?.*(?:\n>.*)*\n?", re.M)


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _citation(policy_id: str, version: int | None) -> str:
    """Versioned policies are citable only with their version.

    Allowing a bare "LEAVE-004" would let a supersession scenario cite the
    policy ambiguously and pass whichever version was retrieved.
    """
    return f"{policy_id}-v{version}" if version is not None else policy_id


def load_handbook(observed_on: date | None = None) -> list[SourceDocument]:
    observed_on = observed_on or date.today()
    docs: list[SourceDocument] = []

    for path in sorted(HANDBOOK_DIR.glob("*.md")):
        if not POLICY_FILENAME.match(path.name):
            continue  # README.md, DEFECTS.md, anything else added later

        raw = path.read_text()
        matter = FRONT_MATTER.match(raw)
        if not matter:
            raise ValueError(f"{path.name}: no front matter")
        meta = yaml.safe_load(matter.group(1)) or {}

        body = raw[matter.end() :]
        # Drop the H1: the title is held as the heading, and repeating it would
        # start every chunk of the policy with the same string.
        body = re.sub(r"^#\s+.*\n", "", body, count=1)
        body = SUPERSEDED_BANNER.sub("", body, count=1)
        text = _clean(body)

        policy_id = meta["policy_id"]
        version = meta.get("version")
        citation = _citation(policy_id, version)

        supersedes = meta.get("supersedes")
        docs.append(
            SourceDocument(
                doc_id=f"company:{citation}",
                citation=citation,
                authority_layer="company",
                # The handbook applies to every employee wherever they work, so
                # it carries no jurisdiction of its own.
                jurisdiction="US",
                section_path=["Meridian Freight Systems", "Employee Handbook"],
                heading=meta["title"],
                text=text,
                content_status="substantive",
                version=version,
                supersedes=f"company:{supersedes}" if supersedes else None,
                effective_from=meta["effective_from"],
                effective_to=meta.get("effective_to"),
                # Authored dates, not inferred from any commencement rule.
                effective_from_is_floor=False,
                observed_on=observed_on,
                source_url="",
                source_note=f"{path.name}; applies_to={meta.get('applies_to', 'unspecified')}",
            )
        )

    return docs
