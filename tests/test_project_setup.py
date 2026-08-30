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


def test_the_local_env_has_no_duplicate_keys() -> None:
    """**A duplicated key in `.env` is silent, and it does not fail where it happens.**

    dotenv keeps the last occurrence. An appended block adding a second provider
    left `OPEN_MODEL_ID=` and `OPEN_MODEL_BASE_URL=` empty, which blanked a
    working Groq configuration, and a Gemini key sat under `OPEN_MODEL_API_KEY`.
    The run then authenticated to Groq with a Google credential and returned
    `Invalid API Key`, which reads as a revoked key rather than as a config
    error two files away.

    CLAUDE.md has said ".env is edited in place, never appended" since the first
    time this happened. Writing the rule down did not stop it happening again,
    which is DL-26: fixing an instance of a bug is not fixing the bug. This is
    the check that does.

    Skipped rather than failed when `.env` is absent, because it is gitignored
    and a clean clone legitimately has none. A test that fails on checkout gets
    deleted, and then it guards nothing.
    """
    import collections

    import pytest

    env = ROOT / ".env"
    if not env.exists():
        pytest.skip(".env is gitignored and absent; nothing local to check")

    counts = collections.Counter(
        line.split("=", 1)[0].strip()
        for line in env.read_text().splitlines()
        if "=" in line and not line.strip().startswith("#")
    )
    duplicated = {k: n for k, n in counts.items() if n > 1}
    assert not duplicated, (
        f"duplicate keys in .env: {duplicated}. dotenv keeps the LAST one, so an "
        "earlier working value is silently discarded. Edit in place; never append."
    )


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

    def without_comments(source: str) -> str:
        """Blank out comments before scanning.

        A comment that quotes the pattern being searched for is not a
        credential. This check flagged its own explanatory note in
        ingest/settings.py, which is the kind of false positive that gets a
        useful check disabled.
        """
        import io
        import tokenize

        lines = source.splitlines(keepends=True)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return source  # unparseable: scan it raw rather than skip it
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            row = token.start[0] - 1
            start_col, end_col = token.start[1], token.end[1]
            line = lines[row]
            lines[row] = line[:start_col] + " " * (end_col - start_col) + line[end_col:]
        return "".join(lines)
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
        for match in pattern.finditer(without_comments(path.read_text())):
            suspicious.append(f"{path.relative_to(ROOT)}: {match.group(1)}")
    assert not suspicious, f"possible hardcoded credentials: {suspicious}"
