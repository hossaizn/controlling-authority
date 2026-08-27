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

TITLE = "Controlling Authority: decision log"

INTRO = """
Every non-obvious choice in this project, with the reasoning behind it. The
failed experiments stay in. So do the diagnoses that turned out wrong, and the
features that were built, measured, and then dropped. A log holding only
successes is a sales document.
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
.wrap { max-width: 820px; }
.nav { margin: 0 0 28px; font-family: var(--mono); font-size: 12.5px; }
.nav a { color: var(--ink); text-decoration-thickness: 2px; text-underline-offset: 3px; }

.toc {
  columns: 2; column-gap: 28px; font-family: var(--mono); font-size: 12.5px;
  line-height: 1.75; list-style: none; padding: 0; margin: 0;
  background: var(--panel); border: 2px solid var(--line); border-radius: 16px;
  padding: 20px 22px; box-shadow: var(--shadow);
}
.toc li { break-inside: avoid; margin-bottom: 4px; }
.toc a { color: var(--ink); text-decoration: none; border-bottom: 1px solid transparent; }
.toc a:hover { border-bottom-color: var(--violet); color: var(--violet); }
@media (max-width: 640px) { .toc { columns: 1; } }

/* Each entry reads as its own card, the way a scenario does on the demo. */
.doc h2 {
  font-size: 26px; margin: 56px 0 6px; padding-top: 28px;
  border-top: 2px solid var(--line);
}
.doc h3 {
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .07em; color: var(--violet);
  margin: 32px 0 10px;
}
.doc p { margin: 14px 0; }
.doc ul, .doc ol { margin: 14px 0; padding-left: 22px; }
.doc li { margin-bottom: 6px; }
.doc strong { font-family: var(--serif); font-size: inherit; text-transform: none;
              letter-spacing: normal; font-weight: 700; }

.doc .tablewrap {
  margin: 20px 0; overflow-x: auto;
  background: var(--panel); border: 2px solid var(--line); border-radius: 14px;
  box-shadow: var(--shadow-sm);
}
.doc table {
  border-collapse: separate; border-spacing: 0; width: 100%;
  font-family: var(--mono); font-size: 12.5px;
}
.doc th, .doc td { padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--line); }
.doc thead th { background: var(--bg); font-weight: 700; text-transform: uppercase;
                letter-spacing: .05em; font-size: 11px; }
.doc tbody tr:last-child td { border-bottom: none; }

.doc pre { margin: 18px 0; }
.doc code {
  font-family: var(--mono); font-size: .88em; background: var(--code);
  border: 1.5px solid var(--line); border-radius: 6px; padding: 1px 6px;
}
.doc pre code { border: none; background: none; padding: 0; }
.doc blockquote {
  margin: 18px 0; padding: 14px 18px; border: 2px solid var(--line);
  border-left-width: 6px; border-left-color: var(--violet);
  border-radius: 12px; background: var(--panel); box-shadow: var(--shadow-sm);
}
.doc blockquote p { margin: 0; }
.doc a { color: var(--violet); text-underline-offset: 3px; }
.doc img { max-width: 100%; }
.doc hr { border: none; border-top: 2px solid var(--line); margin: 40px 0; }
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
    # The scroll container carries the card, not the table. A table that is
    # both the bordered card and the overflow box leaves dead space to the
    # right of any table narrower than the column.
    body = body.replace("<table>", '<div class="tablewrap"><table>')
    body = body.replace("</table>", "</table></div>")

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
  <p class="nav"><a href="/">Back to the demo</a></p>
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
