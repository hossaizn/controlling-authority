"""Vector store tests.

Run against a real Qdrant, in memory. `QdrantClient(":memory:")` is the same
implementation as the server, so filter semantics are genuinely exercised
rather than mocked, and conftest's network block still holds.

Embeddings are deterministic hashes. These tests are about filtering and
storage, not retrieval quality, which Phase 5 measures against real providers.
"""

from __future__ import annotations

from datetime import date

import pytest
from qdrant_client import QdrantClient

from ingest.models import SourceDocument
from retrieval.chunking import chunk_structure_aware
from retrieval.embed import DeterministicEmbeddings
from retrieval.store import ChunkStore, collection_name

OBSERVED = date(2026, 8, 26)


def doc(**overrides) -> SourceDocument:
    base = dict(
        doc_id="us:29-cfr-825.200",
        citation="29 CFR 825.200",
        authority_layer="federal",
        jurisdiction="US",
        section_path=["Part 825", "Subpart B"],
        heading="§ 825.200 Amount of leave.",
        text="An eligible employee may take up to 12 workweeks of leave.",
        content_status="substantive",
        effective_from=date(2016, 12, 1),
        observed_on=OBSERVED,
        source_url="https://example.test",
    )
    base.update(overrides)
    return SourceDocument(**base)


CORPUS = [
    doc(),
    doc(
        doc_id="ca:gov-12945.2",
        citation="Cal. Gov. Code 12945.2",
        authority_layer="state",
        jurisdiction="CA",
        heading="Cal. Gov. Code 12945.2",
        text="An employee with more than 12 months of service may take 12 workweeks.",
        effective_from=date(2023, 1, 1),
    ),
    doc(
        doc_id="ny:wkc-204",
        citation="N.Y. Workers' Comp. Law 204",
        authority_layer="state",
        jurisdiction="NY",
        heading="N.Y. Workers' Comp. Law 204",
        text="Disability and family leave benefits shall be payable.",
        effective_from=date(2016, 4, 8),
    ),
    doc(
        doc_id="company:LEAVE-004-v1",
        citation="LEAVE-004-v1",
        authority_layer="company",
        jurisdiction="US",
        heading="Paid Sick Leave",
        text="Employees accrue up to 24 hours, or 3 days, of paid sick leave per year.",
        version=1,
        effective_from=date(2022, 1, 1),
        effective_to=date(2023, 12, 31),
    ),
    doc(
        doc_id="company:LEAVE-004-v2",
        citation="LEAVE-004-v2",
        authority_layer="company",
        jurisdiction="US",
        heading="Paid Sick Leave",
        text="Employees accrue up to 40 hours, or 5 days, of paid sick leave per year.",
        version=2,
        supersedes="company:LEAVE-004-v1",
        effective_from=date(2024, 1, 1),
    ),
    doc(
        doc_id="oh:absence-family_medical_leave",
        citation="OH (no state provision: family_medical_leave)",
        authority_layer="state",
        jurisdiction="OH",
        heading="No OH state provision: family_medical_leave",
        text="Ohio has no state family and medical leave statute covering private employers.",
        content_status="absent",
        effective_from=date(1900, 1, 1),
    ),
]


@pytest.fixture(scope="module")
def store():
    s = ChunkStore(DeterministicEmbeddings(), strategy="structure", client=QdrantClient(":memory:"))
    s.recreate()
    chunks = [c for d in CORPUS for c in chunk_structure_aware(d)]
    s.index(chunks)
    return s


def _citations(hits):
    return {h.citation for h in hits}


def test_everything_is_indexed(store) -> None:
    hits = store.search("leave", limit=50)
    assert len(hits) == 6


def test_jurisdiction_filter_excludes_other_states(store) -> None:
    """The load-bearing property. A jurisdiction error is a correctness failure,
    not a relevance failure, so this is a filter and never a ranking signal."""
    hits = store.search("leave", jurisdiction="OH", limit=50)
    assert "Cal. Gov. Code 12945.2" not in _citations(hits)
    assert "N.Y. Workers' Comp. Law 204" not in _citations(hits)
    assert all(h.jurisdiction in ("OH", "US") for h in hits)


def test_universal_layers_survive_a_jurisdiction_filter(store) -> None:
    """Federal law and the handbook apply everywhere. Filtering them out would
    leave a state query unable to see the floor it sits on."""
    hits = store.search("leave", jurisdiction="CA", limit=50)
    assert "29 CFR 825.200" in _citations(hits)
    assert "LEAVE-004-v2" in _citations(hits)
    assert "Cal. Gov. Code 12945.2" in _citations(hits)


def test_absence_records_are_retrievable_under_their_jurisdiction(store) -> None:
    """Their entire purpose: an empty result is indistinguishable from a miss."""
    hits = store.search("family leave", jurisdiction="OH", limit=50)
    assert "OH (no state provision: family_medical_leave)" in _citations(hits)


def test_effective_dating_excludes_the_superseded_version(store) -> None:
    """A superseded provision is not a slightly worse answer, it is the wrong
    one."""
    current = _citations(store.search("sick leave", as_of=date(2026, 6, 15), limit=50))
    assert "LEAVE-004-v2" in current
    assert "LEAVE-004-v1" not in current


def test_effective_dating_returns_the_version_in_force_at_the_time(store) -> None:
    past = _citations(store.search("sick leave", as_of=date(2023, 6, 15), limit=50))
    assert "LEAVE-004-v1" in past
    assert "LEAVE-004-v2" not in past


def test_the_changeover_boundary_is_exact(store) -> None:
    """Both sides of 2024-01-01, the day v2 takes effect."""
    last_day = _citations(store.search("sick", as_of=date(2023, 12, 31), limit=50))
    first_day = _citations(store.search("sick", as_of=date(2024, 1, 1), limit=50))
    assert "LEAVE-004-v1" in last_day and "LEAVE-004-v2" not in last_day
    assert "LEAVE-004-v2" in first_day and "LEAVE-004-v1" not in first_day


def test_open_ended_text_is_in_force_far_into_the_future(store) -> None:
    """Text with no end date uses a sentinel upper bound so one range condition
    covers both cases. If that sentinel were wrong, current law would vanish."""
    hits = _citations(store.search("leave", as_of=date(2099, 1, 1), limit=50))
    assert "29 CFR 825.200" in hits
    assert "LEAVE-004-v2" in hits


def test_jurisdiction_and_date_filters_compose(store) -> None:
    hits = store.search("leave", jurisdiction="CA", as_of=date(2023, 6, 15), limit=50)
    cites = _citations(hits)
    assert "Cal. Gov. Code 12945.2" in cites  # effective 2023-01-01
    assert "LEAVE-004-v1" in cites  # in force in 2023
    assert "LEAVE-004-v2" not in cites  # not yet
    assert "N.Y. Workers' Comp. Law 204" not in cites  # wrong jurisdiction


def test_a_date_before_everything_returns_nothing(store) -> None:
    """Honest emptiness. The corpus cannot attest to text before it existed."""
    assert store.search("leave", as_of=date(1800, 1, 1), limit=50) == []


def test_collections_are_separate_per_model_and_strategy() -> None:
    """Vectors from different models are not comparable, and mixing them yields
    plausible nonsense rather than an error."""
    provider = DeterministicEmbeddings()
    assert collection_name(provider, "structure") != collection_name(provider, "fixed")


def test_indexing_the_wrong_strategy_is_refused() -> None:
    from retrieval.chunking import chunk_fixed_size

    s = ChunkStore(DeterministicEmbeddings(), strategy="structure", client=QdrantClient(":memory:"))
    s.recreate()
    with pytest.raises(ValueError, match="refusing to mix"):
        s.index(chunk_fixed_size(doc()))
