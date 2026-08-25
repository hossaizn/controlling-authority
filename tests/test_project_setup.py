"""Guards on the repo itself.

This repo is public and holds keys for four paid APIs. The most likely way a
secret escapes is not a dramatic mistake, it is someone filling in
`.env.example` while testing and committing it without noticing. These tests
make that fail in CI rather than on the internet.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Anything ending in one of these is a credential, not configuration.
SECRET_SUFFIXES = ("_API_KEY", "_SECRET_KEY", "_PUBLIC_KEY")


def _env_example_pairs() -> list[tuple[str, str]]:
    text = (ROOT / ".env.example").read_text()
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        pairs.append((key.strip(), value.strip()))
    return pairs


def test_env_example_has_no_filled_secrets() -> None:
    """Every credential in .env.example must be blank."""
    filled = [k for k, v in _env_example_pairs() if k.endswith(SECRET_SUFFIXES) and v]
    assert not filled, f"credentials must be blank in .env.example: {filled}"


def test_env_is_gitignored() -> None:
    ignored = (ROOT / ".gitignore").read_text()
    assert re.search(r"^\.env$", ignored, re.M), ".env must be gitignored"


def test_env_example_is_not_gitignored() -> None:
    """The negation must survive, or contributors get no template at all."""
    ignored = (ROOT / ".gitignore").read_text()
    assert "!.env.example" in ignored


def test_raw_corpus_is_gitignored() -> None:
    """Raw pulls are large and regenerable. Parsed output is what we commit."""
    ignored = (ROOT / ".gitignore").read_text()
    assert "corpus/raw/" in ignored


def test_no_secret_shaped_literals_in_tracked_python() -> None:
    """Catch keys pasted into source instead of read from the environment.

    Scoped to tracked Python files. Long random-looking strings assigned to a
    variable whose name mentions a key or token are the pattern that matters.
    """
    suspicious: list[str] = []
    # Threshold is 16, not 20. Validated against a real leak in an earlier repo:
    # a 16-character Alpha Vantage key sat under a 20-char threshold undetected
    # while a 40-char Cohere key was caught. Short keys are the ones that slip.
    pattern = re.compile(
        r"""(?ix)
        \b (\w* (?: api_key | secret | token | password ) \w*) \s* = \s*
        ["'] ([A-Za-z0-9_\-]{16,}) ["']
        """
    )
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", "__pycache__", ".git"} for part in path.parts):
            continue
        for match in pattern.finditer(path.read_text()):
            suspicious.append(f"{path.relative_to(ROOT)}: {match.group(1)}")
    assert not suspicious, f"possible hardcoded credentials: {suspicious}"
