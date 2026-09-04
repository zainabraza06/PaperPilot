"""Multi-source search orchestration.

Responsibilities, in order:

1. Interpret the raw query (topic / keyword / identifier / abstract).
2. Fan out to every selected source **concurrently**, each under its own
   wall-clock budget, so one slow provider cannot hold up the response.
3. Turn any failure into a per-source status instead of an exception — a
   search that reaches two of three sources is a successful, degraded
   search, and the UI says so explicitly.
4. Interleave, deduplicate, and return the merged set.

Ranking is deliberately *not* done here; Stage 2 adds a separate ranking
service that consumes this output.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from app.config import Settings
from app.core.errors import (
    SourceParseError,
    SourceRateLimitedError,
    SourceTimeoutError,
    SourceUnavailableError,
)
from app.core.logging import get_logger
from app.models.paper import Paper, SourceName
from app.models.search import (
    IdentifierKind,
    ParsedQuery,
    SearchRequest,
    SearchResponse,
    SourceQuery,
    SourceReport,
    SourceResult,
    SourceStatus,
)
from app.services.dedupe import deduplicate
from app.services.query_parser import parse_query
from app.sources.base import PaperSource
from app.sources.registry import SourceRegistry

logger = get_logger(__name__)


class SearchService:
    """Runs a query across every registered source and merges the results."""

    def __init__(self, registry: SourceRegistry, settings: Settings) -> None:
        self._registry = registry
        self._settings = settings

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        parsed = parse_query(request.query)
        sources = self._registry.select(request.sources)

        logger.info(
            "search intent=%s terms=%r sources=%s",
            parsed.intent.value,
            parsed.search_terms,
            [s.name.value for s in sources],
        )

        results = await asyncio.gather(
            *(self._run_source(source, parsed, request.limit_per_source) for source in sources)
        )

        merged = deduplicate(_interleave(results))
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return SearchResponse(
            query=parsed,
            papers=merged.papers,
            total=len(merged.papers),
            sources=[result.to_report() for result in results],
            elapsed_ms=elapsed_ms,
            duplicates_merged=merged.duplicates_merged,
        )

    # --- per-source execution ---------------------------------------------

    async def _run_source(
        self, source: PaperSource, parsed: ParsedQuery, limit: int
    ) -> SourceResult:
        """Execute one source's part of the fan-out, never raising.

        Every failure mode is mapped onto a ``SourceStatus`` so the caller
        always receives a complete picture of what each provider did.
        """
        started = time.perf_counter()

        def finish(status: SourceStatus, papers: list[Paper], message: str | None) -> SourceResult:
            return SourceResult(
                source=source.name,
                papers=papers,
                status=status,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                message=message,
            )

        try:
            papers = await asyncio.wait_for(
                self._dispatch(source, parsed, limit),
                timeout=self._settings.source_timeout_seconds,
            )
        except TimeoutError:
            logger.warning("%s timed out after %.1fs", source.name.value,
                           self._settings.source_timeout_seconds)
            return finish(
                SourceStatus.TIMEOUT,
                [],
                f"{source.display_name} did not respond within "
                f"{self._settings.source_timeout_seconds:.0f}s",
            )
        except SourceTimeoutError as exc:
            return finish(SourceStatus.TIMEOUT, [], exc.message)
        except SourceRateLimitedError:
            return finish(
                SourceStatus.RATE_LIMITED,
                [],
                f"{source.display_name} is rate limiting requests; try again shortly",
            )
        except SourceParseError as exc:
            logger.warning("%s parse failure: %s", source.name.value, exc.message)
            return finish(SourceStatus.PARSE_ERROR, [], exc.message)
        except SourceUnavailableError as exc:
            return finish(SourceStatus.UNAVAILABLE, [], exc.message)
        except Exception as exc:
            logger.exception("%s raised an unexpected error", source.name.value)
            return finish(SourceStatus.UNAVAILABLE, [], f"unexpected error: {exc}")

        if not papers:
            return finish(SourceStatus.EMPTY, [], f"{source.display_name} returned no matches")
        return finish(SourceStatus.OK, papers, None)

    async def _dispatch(
        self, source: PaperSource, parsed: ParsedQuery, limit: int
    ) -> list[Paper]:
        """Route to an identifier lookup or a relevance search."""
        if not parsed.is_identifier_lookup:
            return await source.search(SourceQuery(parsed=parsed, limit=limit))

        paper = await self._lookup_identifier(source, parsed)
        return [paper] if paper else []

    @staticmethod
    async def _lookup_identifier(source: PaperSource, parsed: ParsedQuery) -> Paper | None:
        """Resolve an identifier through the most direct route each source offers.

        Not every source can answer every identifier — arXiv has no PMID
        index, PubMed has no arXiv index — and "this source cannot answer
        that" is an empty result, not an error.
        """
        identifier = parsed.identifier or ""
        kind = parsed.identifier_kind

        if kind is IdentifierKind.DOI:
            return await source.fetch_by_doi(identifier)

        if kind is IdentifierKind.ARXIV:
            fetch_arxiv = getattr(source, "fetch_by_arxiv_id", None)
            if fetch_arxiv is not None:
                return await fetch_arxiv(identifier)
            # Other sources index arXiv preprints under a DataCite DOI.
            return await source.fetch_by_doi(f"10.48550/arxiv.{identifier}")

        if kind is IdentifierKind.PMID:
            fetch_pmid = getattr(source, "fetch_by_pmid", None)
            if fetch_pmid is not None:
                return await fetch_pmid(identifier)
        return None


def _interleave(results: Sequence[SourceResult]) -> list[Paper]:
    """Round-robin the per-source lists into one.

    Concatenating instead would put one provider's entire result set ahead
    of the others, which biases anything downstream that truncates the list
    — including a user who only reads the first ten rows.
    """
    ordered: list[Paper] = []
    lists = [result.papers for result in results if result.papers]
    for position in range(max((len(items) for items in lists), default=0)):
        for items in lists:
            if position < len(items):
                ordered.append(items[position])
    return ordered


def summarize_sources(reports: Sequence[SourceReport]) -> str:
    """One-line human summary of a fan-out, for logs and CLI output."""
    return ", ".join(
        f"{report.source.value}={report.status.value}({report.returned})" for report in reports
    )


__all__ = ["SearchService", "SourceName", "summarize_sources"]
