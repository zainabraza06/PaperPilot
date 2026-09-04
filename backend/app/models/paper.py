"""Normalized domain models shared by every source connector.

Each external API speaks a different dialect (PubMed XML, arXiv Atom,
Crossref JSON). Connectors are responsible for translating their dialect
into the models defined here, so that nothing downstream of
``app.sources`` ever needs to know where a paper came from.
"""

from __future__ import annotations

import hashlib
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.text import normalize_doi, normalize_title


class SourceName(str, Enum):
    """Identifier for an upstream literature provider."""

    PUBMED = "pubmed"
    ARXIV = "arxiv"
    CROSSREF = "crossref"


class RelevanceScore(BaseModel):
    """Why a paper ranked where it did.

    The components are kept alongside the combined score, not discarded,
    for two reasons: the UI shows a breakdown rather than an unexplained
    number, and an evaluation run can attribute a ranking change to the
    signal that caused it.
    """

    combined: float = Field(ge=0.0, le=1.0, description="Fused score, 0-1, for display.")
    semantic: float = Field(description="Raw cosine similarity of query and paper embeddings.")
    lexical: float = Field(description="Raw BM25 score, unbounded and corpus-relative.")
    rank: int = Field(ge=1, description="1-based position in the ranked result set.")
    strategy: str = Field(description="Fusion strategy that produced `combined`.")


class Author(BaseModel):
    """A paper author.

    ``given``/``family`` are kept separate whenever the source provides them
    because citation formats (BibTeX, RIS) need the split. ``name`` is the
    display form and is always populated.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    given: str | None = None
    family: str | None = None
    affiliation: str | None = None

    @classmethod
    def from_parts(
        cls,
        given: str | None,
        family: str | None,
        affiliation: str | None = None,
    ) -> Author | None:
        """Build an author from name parts, returning ``None`` if both are empty."""
        given = (given or "").strip() or None
        family = (family or "").strip() or None
        if not given and not family:
            return None
        name = " ".join(part for part in (given, family) if part)
        return cls(name=name, given=given, family=family, affiliation=affiliation)

    @classmethod
    def from_display_name(cls, name: str, affiliation: str | None = None) -> Author | None:
        """Build an author from a single display string like ``"Ada Lovelace"``."""
        name = " ".join(name.split())
        if not name:
            return None
        given, _, family = name.rpartition(" ")
        return cls(
            name=name,
            given=given.strip() or None,
            family=family.strip() or None,
            affiliation=affiliation,
        )


class Paper(BaseModel):
    """A single scientific work, normalized across all sources.

    Optionality is deliberate and pervasive: real-world records routinely
    lack an abstract, a DOI, or a usable publication date, and the pipeline
    must degrade gracefully rather than drop those records.
    """

    model_config = ConfigDict(populate_by_name=True)

    # --- identity -------------------------------------------------------
    id: str = Field(description="Stable fingerprint, derived from DOI or source id.")
    doi: str | None = Field(default=None, description="Lowercased, bare DOI (no URL prefix).")
    source: SourceName = Field(description="Provider this record was retrieved from.")
    source_id: str = Field(description="Provider-native identifier (PMID, arXiv id, DOI).")

    # --- bibliographic core ---------------------------------------------
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str | None = None
    published_date: date | None = None
    url: str

    # --- container / venue ----------------------------------------------
    journal: str | None = None
    publisher: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    publication_type: str | None = None

    # --- enrichment ------------------------------------------------------
    keywords: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    citation_count: int | None = None

    # --- ranking (populated by the ranker, absent before Stage 2 runs) ---
    score: RelevanceScore | None = None

    # --- provenance (populated by the deduplicator) ----------------------
    also_found_in: list[SourceName] = Field(
        default_factory=list,
        description="Other sources that returned this same work.",
    )

    @field_validator("doi", mode="before")
    @classmethod
    def _normalize_doi(cls, value: str | None) -> str | None:
        return normalize_doi(value)

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        return " ".join((value or "").split())

    @property
    def year(self) -> int | None:
        return self.published_date.year if self.published_date else None

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.title)

    @property
    def all_sources(self) -> list[SourceName]:
        """Primary source first, then any source that corroborated the record."""
        return [self.source, *self.also_found_in]

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract and self.abstract.strip())

    def completeness(self) -> int:
        """Heuristic field-coverage score used to pick a winner when merging.

        Higher is better. Weighted so that an abstract — the input to the
        whole ranking/summarization pipeline — dominates cosmetic fields.
        """
        score = 0
        score += 10 if self.has_abstract else 0
        score += 4 if self.doi else 0
        score += 3 if self.authors else 0
        score += 2 if self.published_date else 0
        score += 1 if self.journal else 0
        score += 1 if self.pdf_url else 0
        score += 1 if self.keywords else 0
        return score


def make_paper_id(source: SourceName, source_id: str, doi: str | None = None) -> str:
    """Return a stable id for a paper.

    DOI-bearing records share an id across sources so that the same work
    retrieved from PubMed and Crossref collapses naturally. Records without
    a DOI fall back to a source-scoped hash.
    """
    normalized = normalize_doi(doi)
    if normalized:
        return "doi:" + normalized
    digest = hashlib.sha1(f"{source.value}:{source_id}".encode()).hexdigest()[:16]
    return f"{source.value}:{digest}"
