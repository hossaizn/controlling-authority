"""Company handbook ingestion tests."""

from __future__ import annotations

from datetime import date

import pytest

from ingest.company_handbook import load_handbook

OBSERVED = date(2026, 8, 26)


@pytest.fixture(scope="module")
def docs():
    return load_handbook(observed_on=OBSERVED)


def by_citation(docs, citation: str):
    return next(d for d in docs if d.citation == citation)


def test_the_defects_file_is_never_ingested(docs) -> None:
    """DEFECTS.md is ground truth: it states which handbook clauses are wrong
    and what the correct resolution is.

    Ingesting it would put the answers into the corpus the agent retrieves
    from, so every conflict scenario would be answerable by looking up the
    answer key. This is the single most important property in this module.
    """
    blob = " ".join(d.text for d in docs) + " ".join(d.doc_id for d in docs)
    assert "DEFECTS" not in blob
    assert "Seeded defects" not in blob
    assert "ground truth" not in blob.lower()
    assert not any("defect" in d.doc_id.lower() for d in docs)


def test_the_readme_is_not_ingested(docs) -> None:
    """It describes the corpus rather than being part of it, and it also points
    at DEFECTS.md."""
    assert not any("readme" in d.doc_id.lower() for d in docs)
    assert not any("synthetic handbook" in d.text.lower() for d in docs)


def test_every_policy_is_loaded(docs) -> None:
    """Eight policies, one of which exists in two versions."""
    assert len(docs) == 9


def test_citations_match_the_scenario_ground_truth(docs) -> None:
    cites = {d.citation for d in docs}
    assert "LEAVE-003" in cites
    assert {"LEAVE-004-v1", "LEAVE-004-v2"} <= cites
    # A versioned policy must never be citable without its version, or a
    # supersession scenario could cite it ambiguously and pass either way.
    assert "LEAVE-004" not in cites


def test_layer_and_jurisdiction(docs) -> None:
    assert {d.authority_layer for d in docs} == {"company"}
    # The handbook applies to every employee, so it carries no jurisdiction of
    # its own. US is the widest scope available in the contract.
    assert {d.jurisdiction for d in docs} == {"US"}


def test_supersession_is_linked_by_doc_id(docs) -> None:
    """The superseded slice depends on being able to follow this."""
    v2 = by_citation(docs, "LEAVE-004-v2")
    v1 = by_citation(docs, "LEAVE-004-v1")
    assert v2.supersedes == v1.doc_id
    assert v2.version == 2 and v1.version == 1
    assert v1.supersedes is None


def test_effective_dating_comes_from_front_matter(docs) -> None:
    v1 = by_citation(docs, "LEAVE-004-v1")
    v2 = by_citation(docs, "LEAVE-004-v2")
    assert v1.effective_from == date(2022, 1, 1)
    assert v1.effective_to == date(2023, 12, 31)
    assert v2.effective_from == date(2024, 1, 1)
    assert v2.effective_to is None
    # These are authored dates, not inferred from any commencement rule.
    assert v1.effective_from_is_floor is False


def test_the_two_versions_are_in_force_at_different_times(docs) -> None:
    """The entire supersession slice reduces to this."""
    v1 = by_citation(docs, "LEAVE-004-v1")
    v2 = by_citation(docs, "LEAVE-004-v2")
    assert v1.in_force_on(date(2023, 6, 15)) and not v2.in_force_on(date(2023, 6, 15))
    assert v2.in_force_on(date(2026, 6, 15)) and not v1.in_force_on(date(2026, 6, 15))
    # The changeover boundary, both sides.
    assert v1.in_force_on(date(2023, 12, 31)) and not v2.in_force_on(date(2023, 12, 31))
    assert v2.in_force_on(date(2024, 1, 1)) and not v1.in_force_on(date(2024, 1, 1))


def test_front_matter_is_not_left_in_the_body(docs) -> None:
    assert not any("policy_id:" in d.text for d in docs)
    assert not any(d.text.lstrip().startswith("---") for d in docs)


def test_heading_is_the_policy_title(docs) -> None:
    d = by_citation(docs, "LEAVE-002")
    assert d.heading == "Parental Leave"
    assert d.heading not in d.section_path


def test_the_superseded_banner_is_not_treated_as_policy_text(docs) -> None:
    """v1 opens with an editorial note saying it has been superseded. That is
    metadata about the document, and retrieving it as though it were a term of
    the policy would be misleading."""
    v1 = by_citation(docs, "LEAVE-004-v1")
    assert "Superseded." not in v1.text
    assert "See version 2" not in v1.text
    assert "one hour of paid sick leave for every 30 hours" in v1.text


def test_all_policies_carry_text_and_a_hash(docs) -> None:
    assert all(d.content_status == "substantive" for d in docs)
    assert all(d.text.strip() for d in docs)
    assert len({d.content_hash for d in docs}) == 9


def test_every_handbook_citation_used_by_scenarios_resolves(docs) -> None:
    """Closes the loop: the loader checks citations against the filenames, this
    checks them against what the adapter actually produces."""
    from eval.scenarios.loader import load_all

    available = {d.citation for d in docs}
    used = set()
    for s in load_all():
        for c in s.required_citations + s.forbidden_citations + s.must_address:
            if c.startswith("LEAVE-"):
                used.add(c)
    assert used <= available, f"cited but not ingested: {sorted(used - available)}"


def test_the_allowlist_rejects_answer_key_filenames() -> None:
    """Review defeated the first version, which was case-insensitive and let
    LEAVE-009-defects.md through. The earlier test only grepped documents that
    had already loaded, so it never presented a hostile filename at all."""
    from ingest.company_handbook import is_policy_file

    for name in (
        "LEAVE-009-defects.md",
        "LEAVE-001-DEFECTS.md",
        "leave-000-defects.MD",
        "LEAVE-999-ground-truth-answer-key.MD",
        "DEFECTS.md",
        "README.md",
        "notes.md",
    ):
        assert not is_policy_file(name), f"allowlist admitted {name}"


def test_the_allowlist_still_accepts_real_policies() -> None:
    from ingest.company_handbook import is_policy_file

    for name in (
        "LEAVE-001-family-medical-leave.md",
        "LEAVE-004-v1-paid-sick-leave.md",
        "LEAVE-004-v2-paid-sick-leave.md",
    ):
        assert is_policy_file(name), f"allowlist rejected {name}"


def test_answer_key_content_is_refused_even_with_a_valid_filename(tmp_path) -> None:
    """Defence in depth. A file can satisfy the naming convention and still be
    the answer key, and the consequence is silent: an agent retrieving it would
    score well on precisely the scenarios that exist to catch it."""
    import ingest.company_handbook as handbook_mod

    (tmp_path / "LEAVE-010-sneaky.md").write_text(
        "---\npolicy_id: LEAVE-010\ntitle: Sneaky\neffective_from: 2024-01-01\n---\n"
        "# Sneaky\n\nDefect D-1. The correct resolution is that statute controls.\n"
    )
    original = handbook_mod.HANDBOOK_DIR
    handbook_mod.HANDBOOK_DIR = tmp_path
    try:
        with pytest.raises(ValueError, match="answer-key material"):
            handbook_mod.load_handbook(OBSERVED)
    finally:
        handbook_mod.HANDBOOK_DIR = original
