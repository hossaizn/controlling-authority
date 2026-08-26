"""The contract every ingestion adapter produces.

Sources are heterogeneous by nature: a federal XML API, two scraped state sites,
a New York JSON tree, and a Markdown handbook. This is the single shape they all
normalise to, so the chunker and the vector store never learn anything about
where a document came from.

A document is a whole section as published. Splitting happens in Phase 4, which
is why chunking strategy can be compared without re-ingesting anything.

Reworked in Phase 2.5 after review found the first version was federal-shaped in
ways that would not survive the state adapters. See DL-10.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

AuthorityLayer = Literal["federal", "state", "company"]
Jurisdiction = Literal["US", "CA", "NY", "OH"]

# What kind of thing this document is.
#
#   substantive - carries operative text
#   reserved    - exists in the numbering and holds nothing, e.g. "[Reserved]"
#   absent      - a positive record that a layer says nothing on a topic
#
# "absent" is not the same as a document being missing. Ohio has no state family
# leave statute for private employers, and that fact has to be retrievable, or
# the agent cannot tell "no provision exists" from "retrieval failed". It
# therefore carries text like any other document.
ContentStatus = Literal["substantive", "reserved", "absent"]


class SourceDocument(BaseModel):
    # Stable key, ours, used for identity and supersession links.
    # Distinct from `citation`, which is how the source refers to itself and is
    # what appears in an answer. The first version conflated them, which left no
    # way to key two versions of the same handbook policy.
    doc_id: str
    citation: str

    authority_layer: AuthorityLayer
    jurisdiction: Jurisdiction

    # Ancestors only, from the top of the document down to but NOT including
    # this section's own heading. Depth varies by source and must not be
    # assumed: federal is part/subpart, New York's tree is deeper and ragged.
    section_path: list[str]
    heading: str
    text: str

    content_status: ContentStatus = "substantive"

    # Set where a source publishes the same provision in successive versions.
    # The superseded scenario slice depends on being able to follow this link.
    version: int | None = None
    supersedes: str | None = None

    # When this text took effect and when it stopped. `effective_to` of None
    # means still in force as of the snapshot observed.
    effective_from: date
    effective_to: date | None = None

    # True when `effective_from` is the earliest date the source can attest to
    # rather than the date the provision actually began.
    #
    # The eCFR versions feed stamps every section of a part with a shared
    # baseline-load date, so for most sections it reports when the text entered
    # the electronic record, not when it was promulgated. Presenting that as an
    # amendment date would repeat, one level down, the exact error DL-8 was
    # written to avoid. Queries before this date return nothing in force, which
    # is honest: we genuinely cannot attest to the text then.
    effective_from_is_floor: bool = False

    # The date whose snapshot this text was read from. Distinct from
    # effective_from: the snapshot says "this was in force on that day", which
    # is an observation, not the date the provision began.
    observed_on: date

    source_url: str

    # Provenance the source itself prints: Federal Register citations, amendment
    # notes, "Source:" lines. Dropped in the first version, which lost the
    # publication history a reader would need to check a citation.
    source_note: str = ""

    # Detects silent drift in scraped sources between runs. State sites are HTML
    # pages, not APIs, and can change under us without any error surfacing.
    content_hash: str = Field(default="")

    @model_validator(mode="after")
    def _check(self) -> SourceDocument:
        if self.effective_to and self.effective_to < self.effective_from:
            raise ValueError(
                f"{self.doc_id}: effective_to {self.effective_to} precedes "
                f"effective_from {self.effective_from}"
            )
        if self.content_status == "reserved" and self.text.strip():
            raise ValueError(f"{self.doc_id}: reserved section should carry no text")
        if self.content_status != "reserved" and not self.text.strip():
            raise ValueError(f"{self.doc_id}: {self.content_status} document has no text")
        if not self.section_path:
            raise ValueError(f"{self.doc_id}: section_path must not be empty")
        if self.heading in self.section_path:
            raise ValueError(
                f"{self.doc_id}: section_path holds ancestors only; it must not repeat "
                "the document's own heading"
            )
        if self.supersedes and self.supersedes == self.doc_id:
            raise ValueError(f"{self.doc_id}: cannot supersede itself")
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        return self

    def in_force_on(self, day: date) -> bool:
        """Whether this text governed on a given day.

        The query-time filter that makes point-in-time answers possible.
        Inclusive at both ends: a provision is in force on the day it takes
        effect and on the day it is superseded.
        """
        # An absence is a standing fact about a body of law, not a provision
        # with a commencement date. Comparing it against a query date would
        # assert that a jurisdiction's silence "began" on some day, and the
        # sentinel used to express that would then leak into answers.
        if self.content_status == "absent":
            return True
        if day < self.effective_from:
            return False
        return self.effective_to is None or day <= self.effective_to
