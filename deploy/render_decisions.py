"""Render the decision log into a page on the static site.

**This is the artifact, not an appendix.** The demo shows what the system
concludes; the decision log shows why it is built the way it is, including the
experiments that failed, the diagnoses that were wrong, and the features that
were built, measured and then not adopted. Leaving it as a Markdown file in a
repo means a reviewer finds it only if they go looking.

The chrome lives in `deploy/page.py` and is shared with the plan page.
"""

from __future__ import annotations

import re

from deploy import page
from deploy.page import ROOT

LOG = ROOT / "eval" / "decision_log.md"

TITLE = "Controlling Authority: decision log"

INTRO = """
Every non-obvious choice in this project, with the reasoning behind it. The
failed experiments stay in. So do the diagnoses that turned out wrong, and the
features that were built, measured, and then dropped.
"""

# The log opens with its own `# Decision log` heading and a paragraph restating
# what this page already says above the contents. Rendering both puts two
# identical titles and two near-identical intros on one screen. The cut starts
# at the line that is NOT duplicated, so the useful part survives.
BODY_STARTS_AT = "Entries are `DL-n`."


def body_of(markdown: str) -> str:
    if BODY_STARTS_AT not in markdown:
        raise SystemExit(
            f"render_decisions: {BODY_STARTS_AT!r} is not in the log, so the "
            "preamble cut has nothing to anchor on. Fix this rather than "
            "shipping a page with two headings or a missing paragraph."
        )
    return markdown[markdown.index(BODY_STARTS_AT):]


def render() -> str:
    return page.shell(
        title=TITLE,
        heading="Decision log",
        intro=INTRO,
        markdown=body_of(LOG.read_text()),
        current="log",
        footer_note=(
            "Not legal advice. The reasoning here is a record of how this "
            "system was built, not a statement of law."
        ),
    )


def entry_count() -> int:
    return len(re.findall(r"^## DL-\d+", LOG.read_text(), re.M))


if __name__ == "__main__":
    out = ROOT / "dist" / "decisions.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render())
    print(f"wrote {out} ({entry_count()} entries)")
