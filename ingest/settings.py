"""Configuration, read from the environment with `.env` as the local source.

Nothing here holds a default for a credential. A missing key raises at the point
of use with the name of the variable, rather than failing later as an
authentication error that looks like a bug in the caller.

`.env` is loaded once on import. It is gitignored, and `tests/test_project_setup.py`
asserts that every credential in `.env.example` is blank.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def optional(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


# Corpus sources
# Vector store. Not credentials, so they resolve at import.
QDRANT_URL = optional("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = optional("QDRANT_API_KEY")

# Credentials are deliberately NOT mirrored into constants here. An earlier
# version defined NY_SENATE_API_KEY = "NY_SENATE_API_KEY" and similar, which the
# repo's own secret scanner flagged: a variable named *_API_KEY assigned a long
# string literal is indistinguishable from a hardcoded key, whether a human or a
# regex is reading it. Call `require("NY_SENATE_API_KEY")` at the point of use
# instead; it is shorter and reads unambiguously.
