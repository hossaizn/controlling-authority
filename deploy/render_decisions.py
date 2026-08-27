"""Render the decision log into a page on the static site.

**This is the artifact, not an appendix.** The demo shows what the system
concludes; the decision log shows why it is built the way it is, including the
experiments that failed, the diagnoses that were wrong, and the features that
were built, measured and then not adopted. Leaving it as a Markdown file in a
repo means a reviewer finds it only if they go looking.

**Raw HTML in the source is escaped, not passed through** (`html=False`). The
log is content I wrote, so this is not a live threat, but the page is public and
`api/static/index.html` already avoids every HTML-from-data sink for the same
reason. A renderer that passes raw HTML through would be the one place on the
site where that rule does not hold.
"""

from __future__ import annotations

import html as html_lib
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "eval" / "decision_log.md"
PAGE = ROOT / "api" / "static" / "index.html"

TITLE = "Decision log — Controlling Authority"

INTRO = """
Every non-obvious choice in this project, with the reasoning that produced it.
It includes the experiments that failed, the diagnoses that turned out to be
wrong, and the features that were built, measured, and then not adopted. Those
are kept deliberately: a log containing only successes is a sales document.
"""


def stylesheet() -> str:
    """Reuse the demo's CSS so the two pages read as one artifact."""
    match = re.search(r"<style>(.*?)</style>", PAGE.read_text(), re.S)
    if not match:
        raise SystemExit(
            "render_decisions: no <style> block in index.html. The pages would "
            "diverge visually; fix this rather than shipping an unstyled log."
        )
    return match.group(1)


EXTRA_CSS = """
.wrap { max-width: 860px; }
.doc h2 { margin-top: 2.4em; border-top: 1px solid var(--line, #2a2a2a); padding-top: 1.2em; }
.doc h3 { margin-top: 1.8em; }
.doc table { width: 100%; border-collapse: collapse; margin: 1em 0;
             display: block; overflow-x: auto; }
.doc th, .doc td { border: 1px solid var(--line, #2a2a2a); padding: 6px 10px; text-align: left; }
.doc pre { overflow-x: auto; padding: 12px; border-radius: 6px; background: rgba(127,127,127,.12); }
.doc code { font-size: .92em; }
.doc blockquote { margin: 1em 0; padding-left: 1em; border-left: 3px solid var(--line, #2a2a2a); }
.doc img { max-width: 100%; }
.nav { margin: 0 0 24px; }
.toc { columns: 2; font-size: .92em; }
@media (max-width: 640px) { .toc { columns: 1; } }
"""


def anchor_for(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return slug or "section"


def plain(title: str) -> str:
    """Heading text with its inline markup stripped, for the contents list."""
    title = re.sub(r"`([^`]*)`", r"\1", title)
    title = re.sub(r"\*\*([^*]*)\*\*", r"\1", title)
    return re.sub(r"\*([^*]*)\*", r"\1", title)


def headings(markdown: str) -> list[str]:
    return [line[3:].strip() for line in markdown.splitlines() if line.startswith("## ")]


def contents(markdown: str) -> str:
    """A table of contents, because 39 entries is not browsable without one."""
    items = [
        f'<li><a href="#{anchor_for(plain(t))}">{html_lib.escape(plain(t))}</a></li>'
        for t in headings(markdown)
    ]
    return '<ul class="toc">' + "".join(items) + "</ul>"


def add_anchors(rendered: str, markdown: str) -> str:
    """Give every top-level entry an id so the contents list can reach it.

    **Anchored by position, not by matching the text.** Matching on the escaped
    heading string silently missed every heading containing inline code or
    bold, because those render as `<code>`/`<strong>` rather than as the literal
    characters. Six of thirty-nine were skipped, producing contents links that
    scrolled nowhere: a failure invisible except by clicking each one.
    """
    titles = headings(markdown)
    slots = list(re.finditer(r"<h2>", rendered))
    if len(slots) != len(titles):
        raise SystemExit(
            f"render_decisions: {len(titles)} '## ' headings but {len(slots)} "
            "<h2> elements. Positional anchoring is no longer safe; fix this "
            "rather than shipping a contents list with dead links."
        )

    out, last = [], 0
    for title, slot in zip(titles, slots, strict=True):
        out.append(rendered[last:slot.start()])
        out.append(f'<h2 id="{anchor_for(plain(title))}">')
        last = slot.end()
    out.append(rendered[last:])
    return "".join(out)


def render() -> str:
    markdown = LOG.read_text()
    md = MarkdownIt("default", {"html": False, "linkify": False})
    body = add_anchors(md.render(markdown), markdown)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>{stylesheet()}{EXTRA_CSS}</style>
</head>
<body>
<div class="wrap">
  <p class="nav"><a href="/">&larr; Back to the demo</a></p>
  <h1>Decision log</h1>
  <p class="sub">{INTRO.strip()}</p>
  <h2 id="contents">Contents</h2>
  {contents(markdown)}
  <div class="doc">
{body}
  </div>
</div>
</body>
</html>
"""


def entry_count() -> int:
    return len(re.findall(r"^## DL-\d+", LOG.read_text(), re.M))


if __name__ == "__main__":
    out = ROOT / "dist" / "decisions.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render())
    print(f"wrote {out} ({entry_count()} entries)")
