"""Shared chrome for every rendered Markdown page on the static site.

**Lifted from the demo page, never copied.** The nav, the contact block, the
favicon and the stylesheet all come out of `api/static/index.html` at build
time. `agent/build.py` carries the same note about graphs: two assemblies meant
to be identical that drift is a class of bug no test catches, because each one
passes its own.

**Raw HTML in the source is escaped, not passed through** (`html=False`). These
documents are content I wrote, so this is not a live threat, but the pages are
public and `index.html` avoids every HTML-from-data sink for the same reason. A
renderer that passed raw HTML through would be the one place on the site where
that rule does not hold.
"""

from __future__ import annotations

import html as html_lib
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "api" / "static" / "index.html"


def _block(pattern: str, what: str) -> str:
    match = re.search(pattern, PAGE.read_text(), re.S)
    if not match:
        raise SystemExit(
            f"page: no {what} found in index.html. The rendered pages would "
            "ship without it; fix the extraction rather than hand-copying."
        )
    return match.group(0)


def stylesheet() -> str:
    match = re.search(r"<style>(.*?)</style>", PAGE.read_text(), re.S)
    if not match:
        raise SystemExit(
            "page: no <style> block in index.html. The pages would diverge "
            "visually; fix this rather than shipping unstyled documents."
        )
    return match.group(1)


def nav(current: str) -> str:
    """The demo's nav, with the current-page marker moved to `current`."""
    markup = _block(r"<nav class=\"topnav\">.*?</nav>", "nav")
    markup = markup.replace(' class="is-current"', "")
    target = f'data-nav="{current}"'
    if target not in markup:
        raise SystemExit(
            f"page: no nav entry with {target}. The marker would land nowhere "
            "and the page would highlight nothing."
        )
    return markup.replace(target, target + ' class="is-current"', 1)


def favicon() -> str:
    """Lifted, not duplicated. A second copy of a data URI is a second thing to
    forget, and these pages 404ed on /favicon.ico without it."""
    return _block(r'<link rel="icon"[^>]*>', "favicon link")


def contact() -> str:
    return _block(r"<div class=\"contact\">.*?</div>", "contact block")


MAIL_JS = """
<script>
(function () {
  var node = document.getElementById("mail");
  if (!node) return;
  var user = ["hossain", "zulqarnayan"].join("");
  var host = ["gmail", "com"].join(".");
  node.href = "mailto:" + user + "@" + host;
  node.textContent = user + "@" + host;
})();
</script>
"""

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

.doc { margin-top: 34px; }
.doc > :first-child { margin-top: 0; }
.doc h2 {
  font-size: 26px; margin: 56px 0 6px; padding-top: 28px;
  border-top: 2px solid var(--line);
  scroll-margin-top: 70px;
}
.doc h3 {
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .07em; color: var(--violet);
  margin: 32px 0 10px;
}
.doc h4 { font-size: 17px; margin: 24px 0 8px; }
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
/* These documents separate sections with `---` and every heading already
   carries a top border, which drew two rules with a gap between them. The
   heading's border wins, because it stays attached to what it opens. */
.doc hr { display: none; }
"""


# --- headings, contents and anchors -----------------------------------------


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
    items = [
        f'<li><a href="#{anchor_for(plain(t))}">{html_lib.escape(plain(t))}</a></li>'
        for t in headings(markdown)
    ]
    return '<ul class="toc">' + "".join(items) + "</ul>"


def add_anchors(rendered: str, markdown: str) -> str:
    """Give every top-level section an id so the contents list reaches it.

    **Anchored by position, not by matching the text.** Matching on the escaped
    heading string silently missed every heading containing inline code or
    bold, because those render as `<code>`/`<strong>` rather than as the literal
    characters. Six of thirty-nine were skipped on the decision log, producing
    contents links that scrolled nowhere: invisible except by clicking each one.
    """
    titles = headings(markdown)
    slots = list(re.finditer(r"<h2>", rendered))
    if len(slots) != len(titles):
        raise SystemExit(
            f"page: {len(titles)} '## ' headings but {len(slots)} <h2> "
            "elements. Positional anchoring is no longer safe; fix this rather "
            "than shipping a contents list with dead links."
        )
    out, last = [], 0
    for title, slot in zip(titles, slots, strict=True):
        out.append(rendered[last:slot.start()])
        out.append(f'<h2 id="{anchor_for(plain(title))}">')
        last = slot.end()
    out.append(rendered[last:])
    return "".join(out)


def demote(markdown: str) -> str:
    """Push every heading down one level.

    The plan documents open at `#` for a phase and `##` for a task. Rendered
    as-is they would put several `<h1>`s on a page that already has one, and
    the contents would key off the wrong level. Demoting lands phases on `##`
    and tasks on `###`, which is the shape the decision log already uses.
    """
    return re.sub(r"^(#{1,5}) ", r"#\1 ", markdown, flags=re.M)


def wrap_tables(body: str) -> str:
    """The scroll container carries the card, not the table. A table that is
    both the bordered card and the overflow box leaves dead space to the right
    of any table narrower than the column."""
    return body.replace("<table>", '<div class="tablewrap"><table>').replace(
        "</table>", "</table></div>"
    )


def to_html(markdown: str) -> str:
    md = MarkdownIt("default", {"html": False, "linkify": False})
    return wrap_tables(add_anchors(md.render(markdown), markdown))


def shell(*, title: str, heading: str, intro: str, markdown: str,
          current: str, footer_note: str) -> str:
    """One page skeleton, so the pages cannot drift from each other."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{favicon()}
<style>{stylesheet()}{EXTRA_CSS}</style>
</head>
<body>
{nav(current)}
<div class="wrap">
  <h1>{heading}</h1>
  <p class="sub">{intro.strip()}</p>
  <h2 id="contents">Contents</h2>
  {contents(markdown)}
  <div class="doc">
{to_html(markdown)}
  </div>
  <footer>
{contact()}
    <p>{footer_note}</p>
  </footer>
</div>
{MAIL_JS}
</body>
</html>
"""
