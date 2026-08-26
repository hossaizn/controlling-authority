"""Matching a citation in text, correctly.

Its own module because three places need it and two of them already import each
other: `verify` reads the disclaimer from `compose`, and `compose` needs to know
whether it has named a source. Putting the matcher in either created a cycle,
which is the usual sign that it belonged in neither.

The rule it encodes has been got wrong twice in this project, once in each
direction, so it is written down once.
"""

from __future__ import annotations

import re


def mentions(text: str, citation: str) -> bool:
    """Whether `text` names this exact citation, and not a longer one containing it.

    A bare `citation in text` reintroduces DL-12's prefix trap:
    `Cal. Gov. Code 12945` is a prefix of `Cal. Gov. Code 12945.2`, they are
    different statutes, and both are in this corpus.

    Requiring square brackets was the first fix and it was wrong the other way:
    `compose` brackets the controlling provision but names the beaten handbook in
    prose, so a scorer demanding brackets reported naming the beaten source as a
    flat zero. A trailing boundary handles both without dictating how an answer
    is written.
    """
    return re.search(re.escape(citation) + r"(?![\w.\-])", text) is not None


def resolves_to_retrieved(citation: str, retrieved: set[str]) -> bool:
    """Whether a cited string points at text that was actually retrieved.

    An exact match, or a **subsection** of a retrieved provision. The model
    writes `Cal. Gov. Code 12945.2(b)(13)` where `Cal. Gov. Code 12945.2` was
    retrieved, which is a more precise pointer into the same passage rather than
    an invented source. Failing those accounted for 14 of 30 verification
    failures and was discarding correct answers.

    The extension has to open with `(`, so this cannot quietly accept
    `Cal. Gov. Code 12945.2` on the strength of having retrieved
    `Cal. Gov. Code 12945`, which is a different statute.
    """
    if citation in retrieved:
        return True
    return any(
        citation.startswith(r) and citation[len(r):].lstrip().startswith("(")
        for r in retrieved
    )
