"""Multi-source search orchestration.

Responsibilities, in order:

1. Interpret the raw query (topic / keyword / identifier / abstract).
2. Fan out to every selected source **concurrently**, each under its own
   wall-clock budget, so one slow provider cannot hold up the response.
3. Turn any failure into a per-source status instead of an exception — a
   search that reaches two of three sources is a successful, degraded
   search, and the UI says so explicitly.
4. Interleave, deduplicate, rank, and enrich (entities + clusters).

Every stage after retrieval is injected and optional, and each reports its
own outcome. A search that retrieved papers but could not rank, or could
not cluster, still returns those papers with a reason attached — the UI
distinguishes "no clusters" from "clustering unavailable". Degrading one
stage never costs the user the stages that did work.
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
from app.models.clusters import ClusteringReport, TopicCluster
from app.models.paper import Paper, SourceName
from app.models.search import (
    EnrichmentReport,
    IdentifierKind,
    ParsedQuery,
    RankingReport,
    SearchRequest,
    SearchResponse,
    SourceQuery,
    SourceReport,
    SourceResult,
    SourceStatus,
)
from app.services.dedupe import deduplicate
from app.services.enrichment.clustering import TopicClusterer
from app.services.enrichment.entities import EntityExtractor
from app.services.query_parser import parse_query
from app.services.ranking.hybrid import HybridRanker
from app.sources.base import PaperSource, SupportsArxivLookup, SupportsPmidLookup
from app.sources.registry import SourceRegistry

logger = get_logger(__name__)


class SearchService:
    """Runs a query across every registered source and merges the results."""

    def __init__(
        self,
        registry: SourceRegistry,
        settings: Settings,
        ranker: HybridRanker | None = None,
        entity_extractor: EntityExtractor | None = None,
        clusterer: TopicClusterer | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._ranker = ranker
        self._entity_extractor = entity_extractor
        self._clusterer = clusterer

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
        papers, ranking = await self._rank(parsed, merged.papers)

        # Entity extraction and clustering are independent of each other and
        # both CPU-bound, so they run concurrently in worker threads.
        (papers, entities), (clusters, clustering) = await asyncio.gather(
            self._extract_entities(papers),
            self._cluster(papers),
        )
        _assign_clusters(papers, clusters)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return SearchResponse(
            query=parsed,
            papers=papers,
            total=len(papers),
            sources=[result.to_report() for result in results],
            elapsed_ms=elapsed_ms,
            duplicates_merged=merged.duplicates_merged,
            ranking=ranking,
            clusters=clusters,
            clustering=clustering,
            entities=entities,
        )

    async def _extract_entities(
        self, papers: list[Paper]
    ) -> tuple[list[Paper], EnrichmentReport]:
        """Attach named entities to each paper, degrading to none on failure."""
        if self._entity_extractor is None or not papers:
            return papers, EnrichmentReport(
                applied=False,
                reason="entity extraction is disabled"
                if self._entity_extractor is None
                else "no papers to enrich",
            )

        started = time.perf_counter()
        try:
            per_paper = await asyncio.to_thread(self._entity_extractor.extract, papers)
        except Exception as exc:
            logger.exception("entity extraction failed")
            return papers, EnrichmentReport(applied=False, reason=f"NER failed: {exc}")

        enriched = []
        total = 0
        for paper, found in zip(papers, per_paper, strict=True):
            copy = paper.model_copy(update={"entities": found})
            total += len(found)
            enriched.append(copy)

        return enriched, EnrichmentReport(
            applied=True,
            model=self._entity_extractor.model_id,
            entities_found=total,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _cluster(
        self, papers: list[Paper]
    ) -> tuple[list[TopicCluster], ClusteringReport]:
        """Group the result set into sub-topics, degrading to none on failure."""
        if self._clusterer is None or not papers:
            return [], ClusteringReport(
                applied=False,
                reason="clustering is disabled"
                if self._clusterer is None
                else "no papers to cluster",
            )
        try:
            result = await asyncio.to_thread(self._clusterer.cluster, papers)
        except Exception as exc:
            logger.exception("clustering failed")
            return [], ClusteringReport(applied=False, reason=f"clustering failed: {exc}")
        return result.clusters, result.report

    async def _rank(
        self, parsed: ParsedQuery, papers: list[Paper]
    ) -> tuple[list[Paper], RankingReport]:
        """Order the merged set by relevance, degrading to retrieval order.

        The ranker is given ``parsed.raw`` — the user's original text,
        including a full pasted abstract — not the distilled keyword string
        that was sent upstream. Preserving the raw text for exactly this is
        why the query parser keeps both.

        A ranking failure is not a search failure: an unranked list of the
        right papers is still useful, so the outcome is reported the same
        way a source failure is.
        """
        if self._ranker is None or not papers:
            return papers, RankingReport(
                applied=False,
                reason="ranking is disabled" if self._ranker is None else "no papers to rank",
            )

        started = time.perf_counter()
        try:
            ranked = await self._ranker.arank(parsed.raw, papers)
        except Exception as exc:
            logger.exception("ranking failed; returning retrieval order")
            return papers, RankingReport(applied=False, reason=f"ranking failed: {exc}")

        return ranked, RankingReport(
            applied=True,
            strategy=self._ranker.strategy.value,
            model=self._ranker.model_id,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
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
            if isinstance(source, SupportsArxivLookup):
                return await source.fetch_by_arxiv_id(identifier)
            # Other sources index arXiv preprints under a DataCite DOI.
            return await source.fetch_by_doi(f"10.48550/arxiv.{identifier}")

        if kind is IdentifierKind.PMID and isinstance(source, SupportsPmidLookup):
            return await source.fetch_by_pmid(identifier)
        return None


def _assign_clusters(papers: list[Paper], clusters: Sequence[TopicCluster]) -> None:
    """Stamp each paper with its cluster id.

    Denormalized onto the paper as well as listed on the cluster because
    the results list renders per-paper and should not have to search every
    cluster's membership to find a badge.
    """
    if not clusters:
        return
    lookup = {
        paper_id: cluster.id for cluster in clusters for paper_id in cluster.paper_ids
    }
    for paper in papers:
        paper.cluster_id = lookup.get(paper.id)


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
