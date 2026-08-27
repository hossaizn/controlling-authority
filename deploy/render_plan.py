"""Render the implementation plan into a page on the static site.

**The decision log refers to phases forty times and never says what they are.**
"Status: decided, Phase 6.3" means nothing to a reader who cannot see the plan
those phases came from, so the log's most-repeated reference was the one thing
the site did not carry.

**The plan and the outcome disagree in places, and that is the point.** Phase 10
here says "single container on Fly.io". Fly retired its free tier partway
through, so the demo shipped as a static site instead, which DL-39 records. A
plan that survived contact with reality unchanged is usually a plan nobody
followed.

Each file's own `# ` title is replaced, because the two documents are chapters
of one plan here rather than two separate papers.
"""

from __future__ import annotations

import re

from deploy import page
from deploy.page import ROOT

PLANS = (
    (ROOT / "docs/superpowers/plans/2026-08-25-controlling-authority.md",
     "Phases 0 to 5: corpus, ground truth, retrieval"),
    (ROOT / "docs/superpowers/plans/2026-08-26-phases-6-to-10.md",
     "Phases 6 to 10: the agent, and shipping it"),
)

TITLE = "Controlling Authority: implementation plan"

# No standfirst. The contents list says what this page is faster than a
# paragraph does, and the footer carries the one thing a reader needs: where the
# plan and the finished system disagree.
INTRO = ""


def combined() -> str:
    """Both documents, retitled and demoted to sit under one page heading."""
    parts = []
    for path, title in PLANS:
        text = path.read_text()
        lines = text.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise SystemExit(
                f"render_plan: {path.name} does not open with a '# ' title, so "
                "the retitle has nothing to replace. Fix this rather than "
                "shipping two competing headings."
            )
        parts.append("\n".join([f"# {title}", *lines[1:]]))
    return page.demote(strip_tooling("\n\n".join(parts)))


# The plan template opens each file with a directive aimed at the agent that
# executes it, not at a reader. Publishing it puts "REQUIRED SUB-SKILL" at the
# top of the page, above the goal. It is not a plan decision and carries nothing
# a reviewer needs, so it is dropped. Everything the plan actually says stays.
TOOLING_DIRECTIVE = re.compile(r"^> \*\*For agentic workers:\*\*.*$", re.M)


def strip_tooling(markdown: str) -> str:
    cleaned, n = TOOLING_DIRECTIVE.subn("", markdown)
    if not n:
        raise SystemExit(
            "render_plan: the tooling directive is gone from the plans. If it "
            "was removed at source, delete this strip rather than leaving a "
            "rule that silently matches nothing."
        )
    return cleaned


def phase_count() -> int:
    return len(re.findall(r"^#+ PHASE ", combined(), re.M))


def render() -> str:
    return page.shell(
        title=TITLE,
        heading="Implementation plan",
        intro=INTRO,
        markdown=combined(),
        current="plan",
        footer_note=(
            "Written before the work, and left as written. Phase 10 says "
            "Fly.io; the demo shipped on Cloudflare Pages after Fly retired "
            "its free tier, which DL-39 records."
        ),
    )


if __name__ == "__main__":
    out = ROOT / "dist" / "plan.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render())
    print(f"wrote {out} ({phase_count()} phases)")
