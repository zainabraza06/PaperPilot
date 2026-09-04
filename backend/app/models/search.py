"""Request/response models for the search pipeline.

These are the contract between the API layer, the search service, and the
connectors. Keeping them separate from ``Paper`` keeps the domain model
free of transport concerns.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.paper import Paper, SourceName


class QueryIntent(str, Enum):
    """What the user appears to have typed.

    The four intents map onto the four input modes the product promises:
    a topic, a keyword, an identifier lookup, or a pasted abstract.
    """

    TOPIC = "topic"
    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    ABSTRACT_SNIPPET = "abstract_snippet"


class IdentifierKind(str, Enum):
    DOI = "doi"
    ARXIV = "arxiv"
    PMID = "pmid"


class ParsedQuery(BaseModel):
    """The user's raw input, interpreted.

    ``search_terms`` is what actually gets sent to the upstream APIs: for a
    pasted abstract that is a distilled keyword string, because no provider
    accepts a paragraph as a query. ``raw`` is preserved untouched because
    Stage 2 embeds the full text for semantic ranking.
    """

    raw: str
    intent: QueryIntent
    search_terms: str
    identifier_kind: IdentifierKind | None = None
    identifier: str | None = None
    keywords: list[str] = Field(default_factory=list)

    @property
    def is_identifier_lookup(self) -> bool:
        return self.intent is QueryIntent.IDENTIFIER


class SourceQuery(BaseModel):
    """A single connector's view of a search request."""

    parsed: ParsedQuery
    limit: int = 20

    @property
    def terms(self) -> str:
        return self.parsed.search_terms


class SourceStatus(str, Enum):
    """Outcome of one source's contribution to a fan-out.

    The frontend renders these directly: a partial result ("no results from
    PubMed, showing arXiv and Crossref only") is a legitimate, informative
    state rather than an error.
    """

    OK = "ok"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    PARSE_ERROR = "parse_error"
    SKIPPED = "skipped"

    @property
    def is_failure(self) -> bool:
        return self not in (SourceStatus.OK, SourceStatus.EMPTY, SourceStatus.SKIPPED)


class SourceReport(BaseModel):
    """Per-source diagnostics returned alongside every search."""

    source: SourceName
    status: SourceStatus
    returned: int = 0
    elapsed_ms: int = 0
    message: str | None = None


class SourceResult(BaseModel):
    """What a connector hands back to the search service."""

    source: SourceName
    papers: list[Paper] = Field(default_factory=list)
    status: SourceStatus = SourceStatus.OK
    elapsed_ms: int = 0
    message: str | None = None

    def to_report(self) -> SourceReport:
        return SourceReport(
            source=self.source,
            status=self.status,
            returned=len(self.papers),
            elapsed_ms=self.elapsed_ms,
            message=self.message,
        )


class RankingReport(BaseModel):
    """Whether relevance ranking ran, and with what.

    Reported for the same reason source failures are: the frontend needs to
    tell a user "these results are in retrieval order, not relevance order"
    rather than silently presenting a worse list as if it were ranked.
    """

    applied: bool
    strategy: str | None = None
    model: str | None = None
    elapsed_ms: int = 0
    reason: str | None = Field(default=None, description="Why ranking did not run.")


class SearchRequest(BaseModel):
    """Inbound search parameters."""

    query: str = Field(min_length=1, max_length=8000)
    limit_per_source: int = Field(default=20, ge=1, le=100)
    sources: list[SourceName] | None = Field(
        default=None,
        description="Restrict the fan-out; defaults to every registered source.",
    )


class SearchResponse(BaseModel):
    """Merged, deduplicated results plus the provenance of the fan-out."""

    query: ParsedQuery
    papers: list[Paper]
    total: int
    sources: list[SourceReport]
    elapsed_ms: int
    duplicates_merged: int = 0
    ranking: RankingReport = Field(default_factory=lambda: RankingReport(applied=False))

    @property
    def degraded(self) -> bool:
        """True when at least one source failed but we still have results."""
        return any(report.status.is_failure for report in self.sources)
