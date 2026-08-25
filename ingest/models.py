"""The contract every ingestion adapter produces.

Sources are heterogeneous by nature: a federal XML API, two scraped state sites,
and a Markdown handbook. This is the single shape they all normalise to, so the
chunker and the vector store never learn anything about where a document came
from.

A document is a whole section as published. Splitting happens in Phase 4, which
is why chunking strategy can be compared without re-ingesting anything.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, model_validator

AuthorityLayer = Literal["federal", "state", "company"]
Jurisdiction = Literal["US", "CA", "NY", "OH"]


class SourceDocument(BaseModel):
    # The citation as the source itself states it, never reconstructed by us.
    # Scenario ground truth is written against these strings, so inventing a
    # format here would fail correct answers on a formatting difference.
    source_id: str
    citation: str

    authority_layer: AuthorityLayer
    jurisdiction: Jurisdiction

    # Hierarchy from the top of the document down to this section, preserved so
    # structure-aware chunking has structure to work with.
    section_path: list[str]
    heading: str
    text: str

    # When this text took effect, and when it stopped. `effective_to` of None
    # means still in force as of the snapshot observed.
    effective_from: date
    effective_to: date | None = None

    # The date whose snapshot this text was read from. Distinct from
    # effective_from: the snapshot says "this was in force on that day", which
    # is an observation, not the date the provision began.
    observed_on: date

    source_url: str

    # A section that exists in the hierarchy but holds no text, e.g. "[Reserved]".
    # Kept rather than dropped so a gap in the numbering is visible as a fact
    # about the corpus instead of looking like a parser failure.
    is_reserved: bool = False

    @model_validator(mode="after")
    def _check(self) -> SourceDocument:
        if self.effective_to and self.effective_to < self.effective_from:
            raise ValueError(
                f"{self.source_id}: effective_to {self.effective_to} precedes "
                f"effective_from {self.effective_from}"
            )
        if not self.is_reserved and not self.text.strip():
            raise ValueError(f"{self.source_id}: substantive section has no text")
        if self.is_reserved and self.text.strip():
            raise ValueError(f"{self.source_id}: reserved section should carry no text")
        if not self.section_path:
            raise ValueError(f"{self.source_id}: section_path must not be empty")
        return self

    def in_force_on(self, day: date) -> bool:
        """Whether this text governed on a given day.

        The query-time filter that makes point-in-time answers possible.
        """
        if day < self.effective_from:
            return False
        return self.effective_to is None or day <= self.effective_to
