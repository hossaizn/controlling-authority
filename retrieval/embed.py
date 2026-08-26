"""Embedding providers, plural on purpose.

DL-1 is open: does a legal-domain embedding model beat a general-purpose one on
statutory text, which is dense with terms of art? It is a plausible hypothesis
and not a fact, so both are implemented behind one interface and neither is
adopted until Phase 5 measures them on the scenario set.

Nothing here picks a winner. The only thing that decides is recall@10.

**Determinism.** Every provider is asked for a stable model version rather than a
floating alias. An embedding index built against a silently updated model is not
comparable with the numbers that justified choosing it, and DL-1's result would
quietly stop meaning anything.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ingest.settings import require


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything needed to reproduce an index."""

    provider: str
    model: str
    dimensions: int

    @property
    def collection_suffix(self) -> str:
        """Distinct storage per model.

        Two models produce vectors that are not comparable, so they cannot share
        a collection. Writing one into the other's index yields nonsense
        similarity rather than an error, which is the kind of failure that looks
        like a bad retrieval strategy for a week.
        """
        # Dimensions are part of the identity. The same model at two widths
        # produces incomparable vectors, so they cannot share a collection.
        name = f"{self.provider}_{self.model}_{self.dimensions}"
        return name.replace("-", "_").replace(".", "_")


class EmbeddingProvider(ABC):
    spec: EmbeddingSpec

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus text for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a question.

        Separate from document embedding because several providers, Voyage
        included, expect an input type and return measurably worse retrieval
        when queries are embedded as documents.
        """


class OpenAIEmbeddings(EmbeddingProvider):
    """General-purpose baseline."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.spec = EmbeddingSpec("openai", model, dimensions)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=require("OPENAI_API_KEY"))
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # dimensions must be sent, not merely stored. Without it, constructing
        # OpenAIEmbeddings(dimensions=512) built a 512-wide collection and then
        # wrote 1536-wide vectors into it, failing on the first real call.
        response = self._get_client().embeddings.create(
            model=self.spec.model, input=texts, dimensions=self.spec.dimensions
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class VoyageEmbeddings(EmbeddingProvider):
    """Legal-domain model. The DL-1 hypothesis under test."""

    def __init__(self, model: str = "voyage-law-2", dimensions: int = 1024):
        self.spec = EmbeddingSpec("voyage", model, dimensions)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import voyageai

            self._client = voyageai.Client(api_key=require("VOYAGE_API_KEY"))
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._get_client().embed(
            texts, model=self.spec.model, input_type="document"
        ).embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._get_client().embed(
            [text], model=self.spec.model, input_type="query"
        ).embeddings[0]


class DeterministicEmbeddings(EmbeddingProvider):
    """Hash-based vectors for tests. No network, no key, no cost.

    Not a semantic model and not pretending to be one: it exists so the storage,
    filtering and retrieval plumbing can be tested exhaustively without spending
    money or depending on a provider being up. Retrieval *quality* is measured
    in Phase 5 against the real providers.
    """

    def __init__(self, dimensions: int = 64):
        self.spec = EmbeddingSpec("deterministic", "sha256", dimensions)

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self.spec.dimensions)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


PROVIDERS = {
    "openai": OpenAIEmbeddings,
    "voyage": VoyageEmbeddings,
    "deterministic": DeterministicEmbeddings,
}


def get_provider(name: str) -> EmbeddingProvider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[name]()


class SparseOnly(EmbeddingProvider):
    """No dense vectors at all: retrieval falls to lexical matching.

    Not a degraded mode. Sparse matching with server-side IDF is a legitimate
    retrieval strategy, and it needs no credentials, so the chunking comparison
    can run before DL-1 is settled. Results from this arm are labelled as their
    own configuration rather than reported as though they were hybrid.

    A one-dimensional dense vector is still written, because a collection needs
    a dense configuration to exist. It is never queried.
    """

    def __init__(self) -> None:
        self.spec = EmbeddingSpec("none", "sparse_only", 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


PROVIDERS["sparse"] = SparseOnly
