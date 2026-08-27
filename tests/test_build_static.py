"""Tests for the static site build.

The deployed artifact is the only thing a reviewer ever sees, and it is produced
by a script that rewrites URLs in a page. That is the DL-12 hazard exactly:
`str.replace()` returns the string unchanged when it finds nothing, so a rewrite
that silently no-ops ships a page fetching URLs that do not exist, and the
failure shows up only in someone else's browser.
"""

from __future__ import annotations

import json
import re

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
from deploy.page import ROOT


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
    log.write_text(
        "# Decision log\n\nPreamble that the page header already carries.\n\n"
        "Entries are `DL-n`.\n\n## DL-1: a heading\n\n"
        "A line with <script>alert(1)</script>.\n"
    )
    monkeypatch.setattr(render_decisions, "LOG", log)

    page = render_decisions.render()
    body = page.split("</style>", 1)[1]
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_a_heading_count_mismatch_fails_rather_than_mislabelling() -> None:
    """Positional anchoring is only safe while the counts agree."""
    from deploy import page as page_mod

    with pytest.raises(SystemExit, match="Positional anchoring"):
        page_mod.add_anchors("<h2>only one</h2>", "## a\n\n## b\n")


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


def test_the_demo_links_to_the_decision_log_without_a_redirect() -> None:
    """Cloudflare Pages strips `.html`, so `/decisions.html` answers 308 and the
    browser pays an extra round trip to reach the same page. Linking the served
    path directly also means the link is pinned: the decision log is the artifact
    this project is about, and a demo that does not reach it hides it."""
    from pathlib import Path

    html = (
        Path(__file__).resolve().parent.parent / "api/static/index.html"
    ).read_text()
    assert '<a href="/decisions">' in html
    assert '"/decisions.html"' not in html


# --- nav and footer, shared by both pages -----------------------------------


def test_both_pages_carry_the_same_nav(site) -> None:
    """The decision log lifts the nav out of index.html rather than keeping a
    copy. Two navs meant to be identical that drift is a class of bug no test
    catches, because each page passes its own."""
    home = (site / "index.html").read_text()
    log = (site / "decisions.html").read_text()
    for page in (home, log):
        assert '<nav class="topnav">' in page
        assert 'href="/decisions"' in page
        assert "github.com/hossaizn/controlling-authority" in page


def test_each_page_marks_itself_as_current(site) -> None:
    """A nav that highlights nothing, or highlights the same entry everywhere,
    stops telling the reader where they are."""
    home = (site / "index.html").read_text()
    log = (site / "decisions.html").read_text()
    assert '<a href="/" data-nav="demo" class="is-current">' in home
    assert '<a href="/decisions" data-nav="log" class="is-current"' in log
    assert '<a href="/decisions" data-nav="log" class="is-current"' not in home

    # Exactly one marker per page. The log lifts the demo's nav, so failing to
    # clear the inherited marker leaves two highlighted at once, which reads as
    # a broken nav rather than a missing one.
    assert home.count('class="is-current"') == 1
    assert log.count('class="is-current"') == 1
    assert '<a href="/" data-nav="demo" class="is-current">' not in log


def test_the_email_is_not_sitting_in_the_html(site) -> None:
    """Assembled in JS so a scraper reading the source finds nothing. If the
    address ever appears literally, this stops being true."""
    for name in ("index.html", "decisions.html"):
        page = (site / name).read_text()
        assert "hossainzulqarnayan@gmail.com" not in page, name
        # The local part alone is enough for a scraper to guess the rest, so
        # neither half may appear whole.
        assert "hossainzulqarnayan" not in page, name
        assert 'id="mail"' in page, name


def test_both_pages_link_to_linkedin(site) -> None:
    for name in ("index.html", "decisions.html"):
        page = (site / name).read_text()
        assert "linkedin.com/in/zulqarnayan-hossain" in page, name


def test_every_off_origin_link_opens_safely(site) -> None:
    """Without noopener the target page gets a handle on this one through
    window.opener."""
    import re

    for name in ("index.html", "decisions.html"):
        page = (site / name).read_text()
        for tag in re.findall(r"<a\b[^>]*https?://[^>]*>", page, re.I):
            assert 'target="_blank"' not in tag or "noopener" in tag, (name, tag)


def test_a_nav_that_lost_its_current_marker_fails_the_render() -> None:
    """Positional trust again: if the decision-log entry changes shape, the
    marker would silently land nowhere."""
    import re
    from unittest.mock import patch

    from deploy import page as page_mod

    broken = re.sub(r'data-nav="log"', 'data-nav="gone"', page_mod.PAGE.read_text())
    with patch.object(page_mod, "PAGE") as fake:
        fake.read_text.return_value = broken
        with pytest.raises(SystemExit, match="marker would land nowhere"):
            page_mod.nav("log")


def test_build_rejects_a_page_that_would_load_off_origin(tmp_path, monkeypatch) -> None:
    """The self-contained guarantee is enforced at build time, not only in a
    test of the source. Deleting the call left every other check passing."""
    import deploy.build_static as bs

    page = tmp_path / "page.html"
    page.write_text(
        '<link rel="stylesheet" href="https://cdn.example.com/a.css">'
        '<script>fetch("/api/scenarios.json");'
        "if (health.live_ask === false) disableAsk();</script>"
    )
    monkeypatch.setattr(bs, "SOURCE_PAGE", page)
    monkeypatch.setattr(bs, "URL_REWRITES", ())

    with pytest.raises(SystemExit, match="off-origin"):
        bs.build(tmp_path / "out")


def test_the_log_page_shows_one_title_not_two(site) -> None:
    """The markdown opens with its own `# Decision log` and a paragraph that
    restates the page header. Rendering the file whole put two identical titles
    and two near-identical intros on one screen."""
    html = (site / "decisions.html").read_text()
    assert html.count("<h1>") == 1
    assert html.count("Decision log</h1>") == 1


def test_the_preamble_cut_keeps_the_identifier_note(site) -> None:
    """The cut drops the duplicated lines and nothing else. The DL-n against
    D-n distinction is the one thing in that preamble a reader needs."""
    html = (site / "decisions.html").read_text()
    assert "Handbook defect identifiers" in html
    assert html.count('<h2 id="dl-') == 39


def test_a_log_without_the_cut_anchor_fails_loudly(tmp_path, monkeypatch) -> None:
    """Slicing on a string that moved would silently swallow real entries."""
    log = tmp_path / "log.md"
    log.write_text("# Decision log\n\nNo anchor line here.\n\n## DL-1: x\n")
    monkeypatch.setattr(render_decisions, "LOG", log)
    with pytest.raises(SystemExit, match="preamble cut"):
        render_decisions.render()


# --- the plan page ----------------------------------------------------------


def test_the_plan_page_ships(site) -> None:
    """The decision log refers to phases forty times and never says what they
    are. Without this page its most-repeated reference points at nothing."""
    plan = site / "plan.html"
    assert plan.exists()
    html = plan.read_text()
    assert html.count("<h1>") == 1
    assert "PHASE 0" in html and "PHASE 10" in html


def test_the_plan_covers_every_phase_the_log_cites(site) -> None:
    """A log citing Phase 7 against a plan that stops at 6 sends the reader
    nowhere, which is the failure this page exists to prevent."""
    import re

    log = (ROOT / "eval" / "decision_log.md").read_text()
    cited = {int(n) for n in re.findall(r"Phase (\d+)", log)}
    plan = (site / "plan.html").read_text()
    covered = {int(n) for n in re.findall(r"PHASE (\d+)", plan)}
    assert cited <= covered, f"log cites phases the plan does not carry: {cited - covered}"


def test_the_plan_page_has_no_dead_contents_links(site) -> None:
    import re

    html = (site / "plan.html").read_text()
    ids = set(re.findall(r'<h2 id="([^"]+)"', html))
    links = set(re.findall(r'<li><a href="#([^"]+)"', html))
    assert links and not (links - ids), f"dead: {sorted(links - ids)}"


def test_the_plan_page_marks_itself_current(site) -> None:
    plan = (site / "plan.html").read_text()
    assert 'data-nav="plan" class="is-current"' in plan
    assert plan.count('class="is-current"') == 1


def test_every_page_carries_the_plan_link(site) -> None:
    for name in ("index.html", "decisions.html", "plan.html"):
        assert 'href="/plan"' in (site / name).read_text(), name


def test_the_plan_does_not_publish_the_handbook_answer_key(site) -> None:
    """`DEFECTS.md` maps which policy carries which seeded defect. The plan
    names the defect CATEGORIES, which is fine, and must not start naming the
    policies, which would let a reader solve the conflict scenarios by reading
    this page instead of watching the agent reason."""
    html = (site / "plan.html").read_text()
    defects = ROOT / "corpus" / "handbook" / "DEFECTS.md"
    if not defects.exists():
        pytest.skip("no DEFECTS.md in this checkout")
    text = defects.read_text()

    # **A bare policy id is not the answer key.** The plan quotes
    # `policy_id: LEAVE-004` once, inside a front-matter schema example labelled
    # Parental Leave, while the real LEAVE-004 is a sick-leave supersession
    # case. Nothing there tells a reader the policy is defective. What must not
    # travel is the pairing: a policy against its fault. So this checks the
    # fault text, which is the part that would let someone solve the conflict
    # scenarios by reading rather than by watching the agent reason.
    faults = re.findall(r"^\*\*Fault:\*\*\s*(.+)$", text, re.M)
    assert faults, "DEFECTS.md changed shape; this guard is no longer checking anything"
    for fault in faults:
        stem = fault.strip()[:45]
        assert stem not in html, f"the plan page carries a seeded fault: {stem!r}"

    for heading in re.findall(r"^## (D-\d+: .+)$", text, re.M):
        assert heading not in html, f"the plan page carries a defect heading: {heading!r}"


def test_the_plan_page_carries_no_credentials(site) -> None:
    """The plan quotes an .env.example block. Every value in it must stay
    blank, the same rule the committed .env.example is held to."""
    html = (site / "plan.html").read_text()
    for line in re.findall(r"^([A-Z][A-Z0-9_]*_KEY)=(.*)$", html, re.M):
        assert not line[1].strip(), f"{line[0]} carries a value on the plan page"


def test_a_plan_file_without_a_title_fails_the_render(tmp_path, monkeypatch) -> None:
    """Both documents get retitled, so a file that lost its heading would have
    nothing replaced and would ship two competing titles."""
    from deploy import render_plan

    bad = tmp_path / "p.md"
    bad.write_text("No heading here.\n\n## Task 1.1: x\n")
    monkeypatch.setattr(render_plan, "PLANS", ((bad, "Retitled"),))
    with pytest.raises(SystemExit, match="does not open with"):
        render_plan.combined()


def test_the_plan_page_drops_the_agent_tooling_directive(site) -> None:
    """Each plan file opens with a line aimed at the agent that executes it.
    Published, it puts 'REQUIRED SUB-SKILL' above the goal, which tells a
    reviewer nothing about the project."""
    html = (site / "plan.html").read_text()
    assert "For agentic workers" not in html
    assert "REQUIRED SUB-SKILL" not in html
    # The plan itself survives the strip.
    assert "PHASE 0" in html and "Goal:" in html


def test_the_tooling_strip_fails_if_it_matches_nothing() -> None:
    """A strip rule that quietly matches nothing is worse than no rule: it
    looks like a guarantee and is not one."""
    from deploy import render_plan

    with pytest.raises(SystemExit, match="silently matches nothing|matches nothing"):
        render_plan.strip_tooling("# A plan\n\nNo directive in here.\n")


def test_no_page_ships_an_em_dash(site) -> None:
    """A standing instruction for everything on this site. Enforced here rather
    than re-checked by hand, because it was missed twice: once on the decision
    log and once on the plan, both times because a source document was written
    before the rule and only rendered afterwards."""
    from deploy.build_static import em_dashes_in

    for path in sorted(site.glob("*.html")):
        assert em_dashes_in(path.read_text()) == 0, (
            f"{path.name} carries an em dash. Fix it in the source document, "
            "not in the renderer."
        )

    # The counter itself, so a version that always returns zero fails here
    # rather than passing every page silently.
    assert em_dashes_in("a \u2014 b") == 1
    assert em_dashes_in("a - b") == 0


def test_the_build_refuses_a_page_carrying_an_em_dash(tmp_path, monkeypatch) -> None:
    """Enforced by the build, not only by this file. A guard living inside its
    own assertion cannot be caught when it is weakened."""
    import deploy.build_static as bs

    page = tmp_path / "page.html"
    page.write_text(
        "<script>fetch(\"/api/scenarios.json\");"
        "if (health.live_ask === false) disableAsk();</script>"
        "<p>a \u2014 b</p>"
    )
    monkeypatch.setattr(bs, "SOURCE_PAGE", page)
    monkeypatch.setattr(bs, "URL_REWRITES", ())
    with pytest.raises(SystemExit, match="em dashes reached"):
        bs.build(tmp_path / "out")
