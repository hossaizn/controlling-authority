"""Embedding provider tests.

Only the deterministic provider is exercised: the others need keys, cost money,
and their behaviour is the vendor's rather than ours. What is tested here is the
contract every provider must satisfy, and the properties that keep DL-1 an
honest comparison.
"""

from __future__ import annotations

import pytest

from retrieval.embed import (
    PROVIDERS,
    DeterministicEmbeddings,
    OpenAIEmbeddings,
    VoyageEmbeddings,
    get_provider,
)


def test_both_candidate_models_are_available_and_neither_is_default() -> None:
    """DL-1 is open. A default would decide by omission what is meant to be
    decided by measurement."""
    assert {"voyage-law", "voyage-general"} <= set(PROVIDERS)
    with pytest.raises(TypeError):
        get_provider()  # type: ignore[call-arg]


def test_specs_pin_a_model_version_not_a_floating_alias() -> None:
    """An index built against a silently updated model is not comparable with
    the numbers that justified choosing it."""
    assert get_provider("voyage-law").spec.model == "voyage-law-2"
    assert get_provider("voyage-general").spec.model == "voyage-2"


def test_the_dl1_arms_differ_only_in_domain_specialisation() -> None:
    """Same generation and same width, so a win is attributable to the domain
    model rather than to a newer or larger one."""
    legal = get_provider("voyage-law").spec
    general = get_provider("voyage-general").spec
    assert legal.dimensions == general.dimensions == 1024
    assert legal.model.endswith("-2") and general.model.endswith("-2")
    assert legal.provider == general.provider


def test_collection_suffix_separates_models() -> None:
    suffixes = {
        get_provider("voyage-law").spec.collection_suffix,
        get_provider("voyage-general").spec.collection_suffix,
        DeterministicEmbeddings().spec.collection_suffix,
    }
    assert len(suffixes) == 3
    # Qdrant collection names dislike dots and dashes.
    assert all("." not in s and "-" not in s for s in suffixes)


def test_deterministic_provider_is_stable_across_calls() -> None:
    provider = DeterministicEmbeddings()
    assert provider.embed_query("hello") == provider.embed_query("hello")
    assert provider.embed_query("hello") != provider.embed_query("goodbye")


def test_vectors_have_the_declared_dimension_and_are_normalised() -> None:
    provider = DeterministicEmbeddings(dimensions=64)
    vector = provider.embed_query("family and medical leave")
    assert len(vector) == 64
    assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-9


def test_embedding_an_empty_batch_returns_an_empty_list() -> None:
    """Guards the batching loop: a provider that raises here turns an empty
    slice into an indexing failure."""
    assert DeterministicEmbeddings().embed_documents([]) == []


def test_document_and_query_embedding_are_separate_entry_points() -> None:
    """Several providers, Voyage included, expect an input type and retrieve
    measurably worse when a query is embedded as a document."""
    for cls in (OpenAIEmbeddings, VoyageEmbeddings, DeterministicEmbeddings):
        assert hasattr(cls, "embed_documents") and hasattr(cls, "embed_query")


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("word2vec")


def test_real_providers_do_not_touch_the_network_until_used() -> None:
    """Construction must stay cheap and keyless, or importing the module in a
    test run would demand credentials."""
    assert OpenAIEmbeddings()._client is None
    assert VoyageEmbeddings()._client is None
