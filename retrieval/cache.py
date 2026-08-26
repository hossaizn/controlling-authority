"""On-disk embedding cache.

Embeddings are pure functions of (model, text), so recomputing them is waste in
every circumstance and an obstacle in two specific ones: an unpaid Voyage
account is capped at three requests a minute, and any re-run of the evaluation
would otherwise pay the full cost again to produce identical vectors.

Cached by content hash rather than by position, so reordering the corpus or
re-chunking part of it reuses everything unchanged.

The model name is part of the key. Two models' vectors are not comparable, and
serving one from the other's cache would produce plausible nonsense rather than
an error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw" / "embeddings"


def _key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()[:24]
    return f"{model.replace('/', '_')}__{digest}"


class EmbeddingCache:
    def __init__(self, model: str):
        self.model = model
        self.dir = CACHE_DIR / model.replace("/", "_")
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, text: str) -> list[float] | None:
        path = self.dir / f"{_key(self.model, text)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put(self, text: str, vector: list[float]) -> None:
        (self.dir / f"{_key(self.model, text)}.json").write_text(json.dumps(vector))

    def get_many(self, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
        """Returns (hits-or-None per text, indices still needing computation)."""
        found = [self.get(t) for t in texts]
        missing = [i for i, v in enumerate(found) if v is None]
        return found, missing
