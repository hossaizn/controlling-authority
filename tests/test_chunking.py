"""Chunking tests.

Values are pinned rather than related. Mutation testing in Phase 2.5 showed that
assertions of the form "a >= b" on derived output survive an entire family of
wrong derivations.
"""

from __future__ import annotations

from datetime import date

import pytest

from ingest.models import SourceDocument
from retrieval.chunking import (
    MAX_CHUNK_CHARS,
    chunk_corpus,
    chunk_fixed_size,
    chunk_structure_aware,
)

OBSERVED = date(2026, 8, 26)


def make_doc(text: str, **overrides) -> SourceDocument:
    base = dict(
        doc_id="us:29-cfr-825.110",
        citation="29 CFR 825.110",
        authority_layer="federal",
        jurisdiction="US",
        section_path=["Part 825", "Subpart A"],
        heading="§ 825.110 Eligible employee.",
        text=text,
        content_status="substantive",
        effective_from=date(2016, 12, 1),
        observed_on=OBSERVED,
        source_url="https://example.test",
    )
    base.update(overrides)
    return SourceDocument(**base)


SUBDIVIDED = (
    "(a) An eligible employee is an employee of a covered employer who:\n\n"
    "(1) Has been employed by the employer for at least 12 months, and\n\n"
    "(2) Has been employed for at least 1,250 hours of service during the "
    "12-month period immediately preceding the leave.\n\n"
    "(b) The determination of whether an employee meets the hours of service "
    "requirement is made according to the principles of the FLSA."
)


def test_metadata_is_carried_onto_every_chunk() -> None:
    """A chunk that loses its jurisdiction or dates cannot be filtered, and
    filtering is what makes this corpus usable at all."""
    doc = make_doc(SUBDIVIDED, jurisdiction="CA", authority_layer="state")
    for chunk in chunk_structure_aware(doc):
        assert chunk.jurisdiction == "CA"
        assert chunk.authority_layer == "state"
        assert chunk.citation == "29 CFR 825.110"
        assert chunk.effective_from == date(2016, 12, 1)
        assert chunk.section_path == ["Part 825", "Subpart A"]


def test_structure_aware_never_splits_mid_subdivision() -> None:
    """The whole premise. "(2) Has been employed..." must not be separated from
    the sentence it qualifies."""
    doc = make_doc(SUBDIVIDED)
    chunks = chunk_structure_aware(doc, target_chars=120)
    for chunk in chunks:
        # Every chunk begins at a subdivision opener, never mid-clause.
        assert chunk.text.startswith("("), chunk.text[:60]
    assert any("1,250 hours" in c.text for c in chunks)


def test_subdivision_continuations_stay_with_their_opener() -> None:
    doc = make_doc(
        "(a) The opening subdivision.\n\n"
        "A continuation paragraph with no opener of its own.\n\n"
        "(b) The next subdivision."
    )
    chunks = chunk_structure_aware(doc, target_chars=10)
    assert len(chunks) == 2
    assert "continuation paragraph" in chunks[0].text
    assert chunks[1].text.startswith("(b)")


def test_short_documents_are_not_split(sample_short=None) -> None:
    doc = make_doc("A single short provision of no great length.")
    assert len(chunk_structure_aware(doc)) == 1
    assert len(chunk_fixed_size(doc)) == 1


def test_oversized_subdivision_is_broken_at_the_hard_limit() -> None:
    """One enormous subdivision must not produce a chunk nothing can match."""
    doc = make_doc("(a) " + "word " * 3000)
    chunks = chunk_structure_aware(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_reserved_documents_produce_no_chunks() -> None:
    """They hold no text. Emitting an empty chunk would put a retrievable
    nothing into the index."""
    doc = make_doc("", content_status="reserved")
    assert chunk_structure_aware(doc) == []
    assert chunk_fixed_size(doc) == []


def test_absence_records_are_chunked_like_any_document() -> None:
    """Absences must be retrievable; that is their entire purpose."""
    doc = make_doc(
        "Ohio has no state family and medical leave statute covering private employers.",
        content_status="absent",
        doc_id="oh:absence-family_medical_leave",
    )
    chunks = chunk_structure_aware(doc)
    assert len(chunks) == 1
    assert chunks[0].content_status == "absent"


def test_fixed_size_windows_overlap() -> None:
    """The baseline is a competent implementation, not a straw man."""
    doc = make_doc("x" * 4000)
    chunks = chunk_fixed_size(doc, size_chars=1000, overlap_chars=200)
    assert len(chunks) == 5
    assert all(len(c.text) <= 1000 for c in chunks)


def test_fixed_size_drops_a_final_window_that_adds_nothing() -> None:
    doc = make_doc("y" * 1100)
    chunks = chunk_fixed_size(doc, size_chars=1000, overlap_chars=200)
    assert len(chunks) == 2
    assert chunks[1].text != chunks[0].text


def test_overlap_must_be_smaller_than_the_window() -> None:
    doc = make_doc("z" * 3000)
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_fixed_size(doc, size_chars=500, overlap_chars=500)


def test_chunk_ids_are_unique_and_name_their_strategy() -> None:
    doc = make_doc(SUBDIVIDED)
    structure = chunk_structure_aware(doc, target_chars=120)
    fixed = chunk_fixed_size(doc, size_chars=120, overlap_chars=20)
    ids = [c.chunk_id for c in structure + fixed]
    assert len(ids) == len(set(ids)), "strategies must not collide in the index"
    assert all("#structure-" in c.chunk_id for c in structure)
    assert all("#fixed-" in c.chunk_id for c in fixed)


def test_embedding_text_carries_context() -> None:
    """A bare subdivision is nearly meaningless alone, and this corpus is full
    of them. Without context, an FMLA chunk and a CFRA chunk read alike."""
    doc = make_doc("(b) The determination is made under the principles of the FLSA.")
    chunk = chunk_structure_aware(doc)[0]
    assert "§ 825.110 Eligible employee." in chunk.embedding_text
    assert "Subpart A" in chunk.embedding_text
    assert chunk.text in chunk.embedding_text
    # The stored text stays clean; only what is embedded gets the prefix.
    assert not chunk.text.startswith("Subpart A")


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_corpus([], strategy="semantic")
