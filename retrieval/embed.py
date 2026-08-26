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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ingest.settings import require
from retrieval.cache import EmbeddingCache
from retrieval.ratelimit import RateBudget, estimate_tokens


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
    """Retained but not used in DL-1.

    The original design compared Voyage against OpenAI, which confounds the
    question being asked. DL-1 asks whether a legal-domain model beats a
    general-purpose one; a cross-vendor comparison also varies the tokenizer,
    the training corpus and the serving stack, so a win could not be attributed
    to domain specialisation. Both arms now run inside Voyage. See DL-18.
    """

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
    """Voyage, either arm of DL-1.

    `voyage-law-2` is the legal-domain model under test. `voyage-2` is the
    general-purpose control: **same generation, same 1024 dimensions**, so the
    only thing varying between arms is domain specialisation.

    Generation matters here. Comparing `voyage-law-2` against `voyage-4` would
    have confounded "legal beats general" with "newer beats older", and a win
    would have been unattributable.
    """

    # Shared across instances: the limit is per account, not per object, so a
    # per-instance budget would let two providers each use the full allowance.
    _budget = RateBudget()

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
        return self._cached(texts, "document")

    def _cached(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Compute only what is not already on disk.

        Re-running the evaluation then costs nothing, which matters most while
        the account is throttled to three requests a minute.
        """
        cache = EmbeddingCache(f"{self.spec.model}_{input_type}")
        found, missing = cache.get_many(texts)
        if missing:
            pending = [texts[i] for i in missing]
            # Wait for room before sending, rather than sending and recovering.
            self._budget.acquire(estimate_tokens(pending))
            fresh = self._with_retry(
                lambda: self._get_client()
                .embed([texts[i] for i in missing], model=self.spec.model,
                       input_type=input_type)
                .embeddings
            )
            for index, vector in zip(missing, fresh, strict=True):
                cache.put(texts[index], vector)
                found[index] = vector
        return [v for v in found if v is not None]

    @staticmethod
    def _with_retry(call, attempts: int = 8):
        """Exponential backoff on rate limits.

        An account without a payment method is capped at 3 requests and 10,000
        tokens per minute. The free token allowance is unaffected, so the limit
        is a pacing constraint rather than a cost one, and waiting is the
        correct response rather than an error.
        """
        import voyageai.error

        delay = 5.0
        for attempt in range(attempts):
            try:
                return call()
            except voyageai.error.RateLimitError:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.7, 90.0)
        raise RuntimeError("unreachable")

    def embed_query(self, text: str) -> list[float]:
        return self._cached([text], "query")[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed many queries in one request.

        The evaluation asks 57 questions per configuration. Sent one at a time
        that is 57 requests, which at three per minute is twenty minutes of
        waiting for work that fits in three calls.
        """
        return self._cached(texts, "query")


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


def voyage_legal() -> VoyageEmbeddings:
    """DL-1 treatment arm: legal-domain."""
    return VoyageEmbeddings(model="voyage-law-2", dimensions=1024)


def voyage_general() -> VoyageEmbeddings:
    """DL-1 control arm: general-purpose, same generation and width."""
    return VoyageEmbeddings(model="voyage-2", dimensions=1024)


PROVIDERS = {
    "voyage-law": voyage_legal,
    "voyage-general": voyage_general,
    "openai": OpenAIEmbeddings,
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
