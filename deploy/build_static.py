"""Build the static site that Cloudflare Pages serves.

**Why static at all.** Everything the demo argues is already pre-computed: the
six curated scenarios, the naive-baseline comparison DL-31 deliberately made
work without a funded key, and the per-node trace. The only part needing a
server is `/api/ask`, which the protection layer caps at single figures a day.
As of 2026 there is no free, no-card, always-on container host left worth
building on, and a server whose one dynamic feature is rate-limited into
irrelevance is not worth a cold start on every reviewer's first click.

**The URL shapes are rewritten, not re-invented.** `/api/scenario/conflict` and
`/api/scenario/conflict/baseline` cannot both exist on a filesystem: one name
would have to be a file and a directory at once. So the static tree mirrors the
layout `api/precomputed` already uses, `<key>.json` and `<key>.baseline.json`,
and the page's fetch calls are rewritten to match.

**Every rewrite asserts it matched, and every rewritten URL is then checked
against the files actually produced.** DL-12: `str.replace()` returns the string
unchanged when it finds nothing, so a rewrite that silently no-ops would ship a
page fetching URLs that do not exist, and the failure would appear only in a
reviewer's browser.

    uv run python -m deploy.build_static            # writes dist/
    uv run python -m deploy.build_static --out /tmp/x
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from api import precomputed
from deploy import render_decisions
from eval.run_retrieval import CORPUS_SNAPSHOT

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PAGE = ROOT / "api" / "static" / "index.html"
DEFAULT_OUT = ROOT / "dist"

# (pattern, replacement). Each MUST match; see the module docstring.
URL_REWRITES: tuple[tuple[str, str], ...] = (
    ('fetch("/api/scenarios")', 'fetch("/api/scenarios.json")'),
    ('fetch("/api/health")', 'fetch("/api/health.json")'),
    (
        'fetch(`/api/scenario/${encodeURIComponent(key)}`)',
        'fetch(`/api/scenario/${encodeURIComponent(key)}.json`)',
    ),
    (
        'fetch(`/api/scenario/${encodeURIComponent(key)}/baseline`)',
        'fetch(`/api/scenario/${encodeURIComponent(key)}.baseline.json`)',
    ),
    ('fetch(`/api/scenario/${k}`)', 'fetch(`/api/scenario/${k}.json`)'),
)

# Cloudflare Pages types a response from the file extension. Without this the
# JSON is served as octet-stream and `r.json()` rejects.
HEADERS = """/api/*
  Content-Type: application/json; charset=utf-8
  Cache-Control: public, max-age=300
"""


def rewrite_page(html: str) -> str:
    for old, new in URL_REWRITES:
        if old not in html:
            raise SystemExit(
                f"build_static: rewrite target not found in index.html: {old!r}\n"
                "The page changed shape. Fix the rewrite rather than shipping a "
                "page that fetches URLs the build does not produce."
            )
        html = html.replace(old, new)
    return html


def scenarios_payload() -> dict:
    records = [precomputed.load(k) for k in precomputed.available()]
    return {
        "scenarios": [
            {
                "key": r.key,
                "scenario_id": r.scenario_id,
                "question": r.question,
                "as_of": r.as_of,
                "employee_context": r.employee_context,
            }
            for r in records
            if r
        ]
    }


def health_payload() -> dict:
    """The same shape the live endpoint returns, with one field added.

    `live_ask` is false here and true from `api/app.py`. The page reads it and
    disables the ask box rather than offering a control that cannot work: a
    reviewer who clicks Ask and gets a network error learns something false
    about the system.
    """
    return {
        "status": "ok",
        "live_ask": False,
        "deployment": "static",
        "limits": None,
        "precomputed_available": precomputed.available(),
        "precomputed_stale": precomputed.stale(
            precomputed.current_provenance(CORPUS_SNAPSHOT)
        ),
        "tracing": False,
        "overall_scores": precomputed.overall_scores(),
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def build(out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    written: list[Path] = []

    page = rewrite_page(SOURCE_PAGE.read_text())
    (out / "index.html").write_text(page)
    written.append(out / "index.html")

    (out / "_headers").write_text(HEADERS)
    written.append(out / "_headers")

    # The decision log is the artifact this project is actually about, so it
    # ships as a page rather than as a file a reviewer has to go find.
    (out / "decisions.html").write_text(render_decisions.render())
    written.append(out / "decisions.html")

    write_json(out / "api" / "scenarios.json", scenarios_payload())
    write_json(out / "api" / "health.json", health_payload())
    written.extend([out / "api" / "scenarios.json", out / "api" / "health.json"])

    stale = set(precomputed.stale(precomputed.current_provenance(CORPUS_SNAPSHOT)))
    for key in precomputed.available():
        record = precomputed.load(key)
        if record is None:
            continue
        payload = dict(record.payload)
        payload["stale"] = key in stale
        target = out / "api" / "scenario" / f"{key}.json"
        write_json(target, payload)
        written.append(target)

        baseline = precomputed.load_baseline(key)
        if baseline is not None:
            target = out / "api" / "scenario" / f"{key}.baseline.json"
            write_json(target, baseline)
            written.append(target)

    verify(out, page)
    return written


def fetched_paths(html: str) -> set[str]:
    """Every literal path the built page fetches.

    Template placeholders are expanded against the curated keys, because a
    build that only checked the literal URLs would miss exactly the per-scenario
    files, which are most of them.
    """
    # The delimiter is captured and matched, rather than excluding `)` from the
    # path. Excluding it meant `fetch(`/api/scenario/${encodeURIComponent(key)}`)`
    # never matched at all, because the first `)` inside the call ended the
    # class and the closing backtick was no longer next. Two of the five fetches
    # on the page were silently unchecked, and the build passed only because a
    # third fetch happened to cover the same files.
    paths: set[str] = set()
    for _quote, raw in re.findall(r'fetch\((["`])(/api/[^"`]*)\1', html):
        if "${" not in raw:
            paths.add(raw)
            continue
        for key in precomputed.available():
            paths.add(re.sub(r"\$\{[^}]+\}", key, raw))
    return paths


def external_loads(html: str) -> list[str]:
    """Off-origin resources the page would FETCH at render time.

    **Anchors are stripped first, and the distinction is the whole point.**
    `<a href>` navigates when a human clicks; it fetches nothing on load, cannot
    block rendering, and cannot fail the page when the far end is down. The nav
    links to GitHub and the footer to LinkedIn, so a check matching `href` on
    every element would ban those while claiming to be about loading.

    Stripping every tag instead of only anchors would pass everything, which is
    why this lives here as testable code rather than inline in a test: a guard
    written loosely inside its own assertion cannot be caught by the suite.
    """
    loading = re.sub(r"<a\b[^>]*>", "", html, flags=re.I)
    found = re.findall(
        r'(?:src|href)\s*=\s*["\']((?:https?:)?//[^"\']+)', loading, re.I
    )
    found += re.findall(r"url\(\s*[\"\']?((?:https?:)?//[^)\"\']+)", html, re.I)
    return found


# The one endpoint with no static equivalent: it is a POST that runs the graph.
# Listed explicitly rather than filtered out by method, so adding a second
# dynamic endpoint fails this build instead of shipping a dead control.
DYNAMIC_ONLY = {"/api/ask"}


def verify(out: Path, html: str) -> None:
    """Every URL the page fetches must exist as a file, and be valid JSON.

    The dynamic endpoint is exempt from the file check but NOT from scrutiny:
    the page must also contain the code that disables it, or this build would
    ship a button whose only behaviour is to fail.
    """
    missing = []
    for path in sorted(fetched_paths(html) - DYNAMIC_ONLY):
        target = out / path.lstrip("/")
        if not target.exists():
            missing.append(path)
            continue
        json.loads(target.read_text())
    if missing:
        raise SystemExit(
            "build_static: the page fetches URLs this build did not produce:\n  "
            + "\n  ".join(missing)
        )

    offsite = external_loads(html)
    if offsite:
        raise SystemExit(
            "build_static: the page would load off-origin resources, which "
            "breaks the self-contained guarantee:\n  " + "\n  ".join(offsite)
        )

    if DYNAMIC_ONLY & fetched_paths(html) and "disableAsk()" not in html:
        raise SystemExit(
            "build_static: the page calls a dynamic endpoint but has no path "
            "that disables it. A static deployment would ship a control whose "
            "only behaviour is a network error."
        )


def main() -> int:
    out = DEFAULT_OUT
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    written = build(out)
    print(f"built {len(written)} files into {out}")
    for path in written:
        print(f"  {path.relative_to(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
