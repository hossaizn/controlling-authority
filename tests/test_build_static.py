"""Tests for the static site build.

The deployed artifact is the only thing a reviewer ever sees, and it is produced
by a script that rewrites URLs in a page. That is the DL-12 hazard exactly:
`str.replace()` returns the string unchanged when it finds nothing, so a rewrite
that silently no-ops ships a page fetching URLs that do not exist, and the
failure shows up only in someone else's browser.
"""

from __future__ import annotations

import json

import pytest

from deploy import render_decisions
from deploy.build_static import (
    DYNAMIC_ONLY,
    URL_REWRITES,
    build,
    fetched_paths,
    health_payload,
    rewrite_page,
    scenarios_payload,
)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    build(out)
    return out


# --- the rewrite must not fail open -----------------------------------------


def test_a_rewrite_whose_target_is_missing_fails_the_build() -> None:
    """The whole point. A silent no-op here ships a broken page."""
    with pytest.raises(SystemExit, match="rewrite target not found"):
        rewrite_page("<html>a page with no fetch calls</html>")


def test_every_rewrite_changes_the_page() -> None:
    """A rule whose replacement equals its target would pass the missing-target
    check while doing nothing."""
    for old, new in URL_REWRITES:
        assert old != new, f"rewrite is a no-op: {old!r}"


def test_the_rewritten_page_has_no_extensionless_api_calls_left() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "api/static/index.html"
    ).read_text()
    rewritten = rewrite_page(source)
    for path in fetched_paths(rewritten) - DYNAMIC_ONLY:
        assert path.endswith(".json"), f"{path} would 404 on a static host"


# --- what the build produces ------------------------------------------------


def test_every_fetched_url_exists_as_a_file(site) -> None:
    html = (site / "index.html").read_text()
    for path in sorted(fetched_paths(html) - DYNAMIC_ONLY):
        assert (site / path.lstrip("/")).exists(), f"page fetches missing {path}"


def test_every_produced_api_file_is_valid_json(site) -> None:
    files = list((site / "api").rglob("*.json"))
    assert files, "no API files were produced"
    for path in files:
        json.loads(path.read_text())


def test_the_scenario_and_baseline_names_cannot_collide(site) -> None:
    """`/api/scenario/conflict` and `/api/scenario/conflict/baseline` would need
    one name to be a file and a directory at once. The flat `<key>.baseline.json`
    layout is what avoids that, so it is pinned."""
    for path in (site / "api" / "scenario").iterdir():
        assert path.is_file(), f"{path.name} is a directory; names collide"


def test_the_headers_file_types_the_api_as_json(site) -> None:
    """Without this Cloudflare serves the files as octet-stream and every
    `r.json()` in the page rejects."""
    headers = (site / "_headers").read_text()
    assert "/api/*" in headers
    assert "application/json" in headers


# --- the static build must not claim it can answer --------------------------


def test_the_static_health_reports_that_live_answering_is_off(site) -> None:
    health = json.loads((site / "api" / "health.json").read_text())
    assert health["live_ask"] is False
    assert health["deployment"] == "static"


def test_the_live_server_reports_that_it_can_answer() -> None:
    """The two deployments differ in data, not in code. If both said the same
    thing the page could not tell them apart."""
    from fastapi.testclient import TestClient

    from api.app import create_app

    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["live_ask"] is True


def test_the_page_disables_the_ask_box_when_answering_is_off(site) -> None:
    html = (site / "index.html").read_text()
    assert "health.live_ask === false" in html
    assert "function disableAsk()" in html


def test_a_page_calling_a_dynamic_endpoint_without_a_disable_path_fails(
    tmp_path,
) -> None:
    """Otherwise the static site ships a button whose only behaviour is a
    network error, which teaches a reviewer that the system is broken."""
    from deploy.build_static import verify

    with pytest.raises(SystemExit, match="disables it"):
        verify(tmp_path, 'fetch("/api/ask", {method: "POST"})')


# --- the numbers on the page come from the committed snapshot ---------------


def test_health_carries_the_measured_scores() -> None:
    scores = health_payload()["overall_scores"]
    assert scores["n"] == 92
    assert 0 < scores["fully_correct"] < 1


def test_every_curated_scenario_is_published() -> None:
    from api import precomputed

    keys = {s["key"] for s in scenarios_payload()["scenarios"]}
    assert keys == set(precomputed.available())


def test_the_conflict_scenario_still_shows_the_argument(site) -> None:
    """The demo exists to show precedence beating the top-ranked passage. If the
    two arms ever agree here, the flagship button silently stops arguing."""
    agent = json.loads((site / "api/scenario/conflict.json").read_text())
    baseline = json.loads((site / "api/scenario/conflict.baseline.json").read_text())
    assert agent["controlling_authority"] != baseline["controlling_authority"]
    assert baseline["correct"] is False


# --- the decision log page --------------------------------------------------


def test_the_decision_log_page_has_no_dead_contents_links(site) -> None:
    """Anchoring by matching heading text skipped every heading containing
    inline code, producing six links that scrolled nowhere."""
    import re

    html = (site / "decisions.html").read_text()
    ids = set(re.findall(r'<h2 id="([^"]+)"', html))
    links = set(re.findall(r'<li><a href="#([^"]+)"', html))
    assert links, "no contents list was rendered"
    assert not (links - ids), f"dead contents links: {sorted(links - ids)}"


def test_the_decision_log_renders_its_tables(site) -> None:
    """Most of the evidence in the log is in tables. A renderer without table
    support would drop them to unreadable pipe soup and lose the results."""
    html = (site / "decisions.html").read_text()
    assert html.count("<table>") > 20


def test_the_decision_log_escapes_rather_than_passes_through_html(
    tmp_path, monkeypatch
) -> None:
    """The page is public, and index.html avoids every HTML-from-data sink for
    the same reason. The renderer must not be the one exception.

    Driven through `render()` against a real file, not through a locally
    constructed parser: the first version built its own MarkdownIt with
    html=False and asserted on that, so flipping the flag at the actual call
    site changed nothing the test could see.
    """
    log = tmp_path / "log.md"
    log.write_text("## DL-1: a heading\n\nA line with <script>alert(1)</script>.\n")
    monkeypatch.setattr(render_decisions, "LOG", log)

    page = render_decisions.render()
    body = page.split("</style>", 1)[1]
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_a_heading_count_mismatch_fails_rather_than_mislabelling() -> None:
    """Positional anchoring is only safe while the counts agree."""
    with pytest.raises(SystemExit, match="Positional anchoring"):
        render_decisions.add_anchors("<h2>only one</h2>", "## a\n\n## b\n")


# --- the build's own checks must be reachable from the build ----------------


def test_build_rejects_a_page_that_fetches_something_it_did_not_produce(
    tmp_path, monkeypatch
) -> None:
    """`build()` calls `verify()` last. Deleting that call left every check in
    this file still passing, because they all re-ran the verification
    themselves rather than exercising the build's use of it."""
    import deploy.build_static as bs

    page = tmp_path / "page.html"
    page.write_text(
        '<script>fetch("/api/scenarios");'
        'fetch("/api/nonexistent");'
        "if (health.live_ask === false) disableAsk();</script>"
    )
    monkeypatch.setattr(bs, "SOURCE_PAGE", page)
    monkeypatch.setattr(bs, "URL_REWRITES", ())

    with pytest.raises(SystemExit, match="did not produce"):
        bs.build(tmp_path / "out")


def test_template_fetches_expand_to_every_scenario() -> None:
    """A `fetch(`/api/scenario/${key}`)` covers most of the files on the site.
    Returning only the literal URLs made the coverage check pass while
    inspecting almost nothing."""
    from api import precomputed

    paths = fetched_paths('fetch(`/api/scenario/${encodeURIComponent(key)}.json`)')
    assert paths == {
        f"/api/scenario/{k}.json" for k in precomputed.available()
    }
    assert len(paths) > 1
