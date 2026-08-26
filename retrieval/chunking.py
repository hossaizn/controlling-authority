"""Chunking, in two strategies so they can be compared rather than assumed.

DL-1 asks which embedding model suits a regulatory corpus. This module poses the
same question about splitting: structure-aware against a fixed-size baseline,
measured on the scenario set in Phase 5 before either is adopted.

Both satisfy one interface so the eval can swap them, and both carry the parent
document's metadata forward untouched. A chunk that loses its jurisdiction or
its dates cannot be filtered, and filtering is what makes this corpus usable.

**Why structure might win.** Regulatory text is written in numbered subdivisions
and a fixed window cuts through them, so "(2) Has been employed for at least
1,250 hours" can end up in a different chunk from the sentence that governs it.

**Why it might not.** The corpus is lopsided: the longest federal section runs
30,111 characters and the shortest 231. Structure-aware splitting inherits that
spread, and very long chunks dilute an embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingest.models import SourceDocument

# Subdivision openers as the sources actually write them:
#   (a) / (1) / (a)(1) at the start of a block
SUBDIVISION = re.compile(r"^\((?:[a-z]{1,2}|\d{1,2}|[ivx]{1,4})\)", re.I)

DEFAULT_TARGET_CHARS = 1_500
DEFAULT_OVERLAP_CHARS = 200
MAX_CHUNK_CHARS = 4_000


@dataclass(frozen=True)
class Chunk:
    """A retrievable span, carrying everything needed to filter and cite it."""

    chunk_id: str
    doc_id: str
    citation: str
    authority_layer: str
    jurisdiction: str
    section_path: list[str]
    heading: str
    content_status: str
    text: str
    ordinal: int
    strategy: str
    version: int | None = None
    supersedes: str | None = None
    effective_from: object = None
    effective_to: object = None
    effective_from_is_floor: bool = False
    source_url: str = ""
    source_note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded.

        The heading and the last hierarchy level are prepended because a bare
        subdivision like "(b) The determination is made by..." is nearly
        meaningless on its own, and the corpus is full of them. Without this a
        chunk from FMLA and a chunk from CFRA can read almost identically.
        """
        parent = self.section_path[-1] if self.section_path else ""
        context = " > ".join(part for part in (parent, self.heading) if part)
        return f"{context}\n\n{self.text}" if context else self.text


def _emit(doc: SourceDocument, pieces: list[str], strategy: str) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{strategy}-{i}",
            doc_id=doc.doc_id,
            citation=doc.citation,
            authority_layer=doc.authority_layer,
            jurisdiction=doc.jurisdiction,
            section_path=list(doc.section_path),
            heading=doc.heading,
            content_status=doc.content_status,
            text=piece,
            ordinal=i,
            strategy=strategy,
            version=doc.version,
            supersedes=doc.supersedes,
            effective_from=doc.effective_from,
            effective_to=doc.effective_to,
            effective_from_is_floor=doc.effective_from_is_floor,
            source_url=doc.source_url,
            source_note=doc.source_note,
        )
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]


def _pack(units: list[str], target: int, hard_max: int) -> list[str]:
    """Greedily combine units up to a target, splitting any that exceed the max.

    Keeps small sections whole rather than emitting a chunk per sentence, and
    stops one enormous subdivision from producing a chunk nothing can match.
    """
    packed: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > hard_max:
            if current:
                packed.append(current)
                current = ""
            for start in range(0, len(unit), hard_max):
                packed.append(unit[start : start + hard_max])
            continue
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) > target and current:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def chunk_structure_aware(
    doc: SourceDocument,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Split on the document's own subdivisions, never mid-subdivision."""
    if doc.content_status == "reserved":
        return []
    blocks = [b.strip() for b in doc.text.split("\n\n") if b.strip()]

    # Group each subdivision opener with the blocks that continue it, so "(2)"
    # stays attached to the sentence it belongs to.
    units: list[str] = []
    for block in blocks:
        if SUBDIVISION.match(block) or not units:
            units.append(block)
        else:
            units[-1] = f"{units[-1]}\n\n{block}"

    # Not every source is written in numbered subdivisions. The nine handbook
    # policies are prose under Markdown headings and contain no "(a)" markers at
    # all, so the loop above folded each entire policy into a single unit and
    # target_chars became inert: a 1,612 character policy emitted one oversized
    # chunk while the fixed-size baseline correctly emitted two.
    #
    # With no structure to respect, fall back to paragraphs, which is the
    # document's own structure at the next level down. Doing nothing here would
    # have made the strategy comparison meaningless for the entire company layer.
    if len(units) == 1 and len(blocks) > 1:
        units = blocks

    return _emit(doc, _pack(units, target_chars, max_chars), "structure")


def chunk_fixed_size(
    doc: SourceDocument,
    size_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """The baseline to beat: a sliding window that ignores structure entirely.

    Deliberately a competent version rather than a straw man. It overlaps, which
    is what a careful implementation would do, so the comparison in Phase 5 is
    against the obvious approach done properly.
    """
    if doc.content_status == "reserved":
        return []
    text = doc.text
    if len(text) <= size_chars:
        return _emit(doc, [text], "fixed")

    step = size_chars - overlap_chars
    if step <= 0:
        raise ValueError("overlap must be smaller than the window")

    # Redundancy is decided by position, not by content. Testing whether the
    # last window is a substring of the previous one looks equivalent and is
    # not: on repetitive text any short tail is a substring of what precedes it,
    # so real content gets dropped. Statutory text repeats itself constantly.
    pieces: list[str] = []
    previous_end = 0
    for start in range(0, len(text), step):
        end = min(start + size_chars, len(text))
        if pieces and end <= previous_end:
            break  # this window and every later one add nothing new
        pieces.append(text[start:end])
        previous_end = end
    return _emit(doc, pieces, "fixed")


STRATEGIES = {"structure": chunk_structure_aware, "fixed": chunk_fixed_size}


def chunk_corpus(docs: list[SourceDocument], strategy: str = "structure") -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    chunker = STRATEGIES[strategy]
    return [chunk for doc in docs for chunk in chunker(doc)]
