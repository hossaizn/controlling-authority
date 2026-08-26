"""Sparse term vectors, for the half of retrieval embeddings are bad at.

The spec calls for hybrid search because statutory text is dense with terms of
art that carry exact meaning: `FMLA`, `825.200`, `1,250 hours`, `Section 125`.
A dense embedding places those near their paraphrases, which is the opposite of
what a citation lookup needs. Sparse matching finds the literal token.

Weighting is raw term frequency, and IDF is left to Qdrant's `Modifier.IDF`,
which computes it across the collection server-side. Computing IDF here would
mean recomputing it whenever the corpus changed, and getting it silently wrong
whenever it was not recomputed.

Tokens keep internal dots and hyphens so `825.200` survives as one token rather
than becoming `825` and `200`, which would match every section number in the
part.
"""

from __future__ import annotations

import re
from collections import Counter

# Alphanumeric runs, allowing internal . - / so citations and hour thresholds
# stay intact: 825.200, 1,250 is handled by stripping the comma first.
TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")

# One-character tokens carry no retrieval signal and inflate every vector.
MIN_TOKEN_LENGTH = 2


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower().replace(",", "")) if len(t) >= MIN_TOKEN_LENGTH]


def sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """(indices, values) as Qdrant expects.

    Indices are a stable 32-bit hash of the token. `hash()` is unusable here for
    the same reason it was unusable for point ids: Python randomises it per
    process, so the same token would occupy a different dimension on every run
    and a query would never match anything indexed earlier.
    """
    counts = Counter(tokenize(text))
    indices: list[int] = []
    values: list[float] = []
    for token, count in counts.items():
        indices.append(_token_index(token))
        values.append(float(count))
    return indices, values


def _token_index(token: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")
