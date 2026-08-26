"""Qdrant storage and retrieval.

The reason this project uses a real vector database rather than an in-memory
index is filtering. A jurisdiction error is a correctness failure, not a
relevance failure: Ohio law must never surface for a California question, no
matter how similar the text. So jurisdiction and effective dates are applied as
**hard filters before scoring**, never as ranking signals.

Effective dating is filtered the same way. A superseded provision is not a
slightly worse answer, it is the wrong one.

One collection per (embedding model, chunking strategy). Vectors from different
models are not comparable, and mixing them produces plausible nonsense rather
than an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from qdrant_client import QdrantClient, models

from ingest.settings import QDRANT_API_KEY, QDRANT_URL
from retrieval.chunking import Chunk
from retrieval.embed import EmbeddingProvider

# The handbook applies everywhere, so a state query must still see it. Federal
# law applies everywhere too. A jurisdiction filter therefore admits the
# jurisdiction asked about plus the universal layers, and nothing else.
UNIVERSAL_JURISDICTION = "US"


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    citation: str
    authority_layer: str
    jurisdiction: str
    content_status: str
    heading: str
    text: str
    score: float


def collection_name(provider: EmbeddingProvider, strategy: str) -> str:
    return f"ca_{provider.spec.collection_suffix}_{strategy}"


class ChunkStore:
    def __init__(
        self,
        provider: EmbeddingProvider,
        strategy: str = "structure",
        url: str | None = None,
        client: QdrantClient | None = None,
    ):
        self.provider = provider
        self.strategy = strategy
        self.collection = collection_name(provider, strategy)
        self.client = client or QdrantClient(
            url=url or QDRANT_URL, api_key=QDRANT_API_KEY or None
        )

    def recreate(self) -> None:
        """Drop and rebuild the collection, with payload indexes.

        Payload indexes exist so filtering happens inside Qdrant rather than by
        over-fetching and discarding, which silently degrades recall: a query
        filtered after the fact can return fewer than k results with no error.
        """
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.provider.spec.dimensions, distance=models.Distance.COSINE
            ),
        )
        for field, schema in [
            ("jurisdiction", models.PayloadSchemaType.KEYWORD),
            ("authority_layer", models.PayloadSchemaType.KEYWORD),
            ("content_status", models.PayloadSchemaType.KEYWORD),
            ("citation", models.PayloadSchemaType.KEYWORD),
            ("effective_from_ord", models.PayloadSchemaType.INTEGER),
            ("effective_to_ord", models.PayloadSchemaType.INTEGER),
        ]:
            self.client.create_payload_index(self.collection, field, schema)

    def index(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        wrong = [c.chunk_id for c in chunks if c.strategy != self.strategy]
        if wrong:
            raise ValueError(
                f"collection holds {self.strategy!r} chunks; refusing to mix in "
                f"{len(wrong)} from another strategy"
            )

        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.provider.embed_documents([c.embedding_text for c in batch])
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=abs(hash(c.chunk_id)) % (2**63),
                        vector=vector,
                        payload=_payload(c),
                    )
                    for c, vector in zip(batch, vectors, strict=True)
                ],
            )
            written += len(batch)
        return written

    def search(
        self,
        query: str,
        jurisdiction: str | None = None,
        as_of: date | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        must: list[models.Condition] = []

        if jurisdiction:
            # Hard filter, never a ranking signal. Admits the jurisdiction
            # asked about plus the layers that apply everywhere.
            must.append(
                models.FieldCondition(
                    key="jurisdiction",
                    match=models.MatchAny(any=[jurisdiction, UNIVERSAL_JURISDICTION]),
                )
            )

        if as_of:
            ordinal = as_of.toordinal()
            must.append(
                models.FieldCondition(
                    key="effective_from_ord", range=models.Range(lte=ordinal)
                )
            )
            # effective_to is sentinel-high for text still in force, so one
            # range condition covers both cases without a null branch.
            must.append(
                models.FieldCondition(
                    key="effective_to_ord", range=models.Range(gte=ordinal)
                )
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=self.provider.embed_query(query),
            query_filter=models.Filter(must=must) if must else None,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                chunk_id=p.payload["chunk_id"],
                citation=p.payload["citation"],
                authority_layer=p.payload["authority_layer"],
                jurisdiction=p.payload["jurisdiction"],
                content_status=p.payload["content_status"],
                heading=p.payload["heading"],
                text=p.payload["text"],
                score=p.score,
            )
            for p in response.points
        ]


# Text still in force needs an upper bound a range filter can compare against.
# date.max is used rather than null so "currently in force" and "expired" are one
# comparison instead of two code paths.
_OPEN_ENDED = date.max.toordinal()


def _payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "citation": chunk.citation,
        "authority_layer": chunk.authority_layer,
        "jurisdiction": chunk.jurisdiction,
        "content_status": chunk.content_status,
        "heading": chunk.heading,
        "section_path": chunk.section_path,
        "text": chunk.text,
        "ordinal": chunk.ordinal,
        "strategy": chunk.strategy,
        "version": chunk.version,
        "supersedes": chunk.supersedes,
        "effective_from_ord": chunk.effective_from.toordinal(),
        "effective_to_ord": (
            chunk.effective_to.toordinal() if chunk.effective_to else _OPEN_ENDED
        ),
        "source_url": chunk.source_url,
    }
