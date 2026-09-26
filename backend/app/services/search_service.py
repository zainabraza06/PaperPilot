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
from app.models.entities import Entity
from app.models.paper import Paper, SourceName
from app.models.search import (
    CacheReport,
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
from app.models.summary import Summary, SummaryReport
from app.services.dedupe import deduplicate
from app.services.enrichment.clustering import TopicClusterer
from app.services.enrichment.entities import EntityExtractor
from app.services.query_parser import parse_query
from app.services.ranking.hybrid import HybridRanker
from app.services.summarization.summarizer import PaperSummarizer
from app.sources.base import PaperSource, SupportsArxivLookup, SupportsPmidLookup
from app.sources.registry import SourceRegistry
from app.storage.paper_store import PaperStore
from app.storage.search_cache import SearchCache, cache_key

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
        summarizer: PaperSummarizer | None = None,
        paper_store: PaperStore | None = None,
        search_cache: SearchCache | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._ranker = ranker
        self._entity_extractor = entity_extractor
        self._clusterer = clusterer
        self._summarizer = summarizer
        self._paper_store = paper_store
        self._search_cache = search_cache

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        parsed = parse_query(request.query)
        sources = self._registry.select(request.sources)

        # Bound once so the None-check narrows for the whole block: a
        # cache hit can only exist when there is a cache to hit.
        cache = self._search_cache
        key = self._cache_key(request)
        if cache is not None and key is not None:
            cached = await self._cached(key)
            if cached is not None:
                response, age = cached
                # elapsed_ms is rewritten to what *this* request actually
                # took. Replaying the original seven seconds would make the
                # cache invisible in exactly the panel built to show where
                # the time goes.
                response.elapsed_ms = int((time.perf_counter() - started) * 1000)
                response.cache = CacheReport(
                    hit=True, age_seconds=age, ttl_seconds=cache.ttl_seconds
                )
                logger.info("search served from cache (age %ss)", age)
                return response

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

        # The three enrichment passes are independent, and one of them is
        # network-bound while the others are CPU-bound, so they overlap well.
        # Each returns its own product rather than a mutated paper list, so
        # merging them afterwards is a single unambiguous pass.
        (
            (entity_lists, entities),
            (clusters, clustering),
            (summaries, summary_report),
        ) = await asyncio.gather(
            self._extract_entities(papers),
            self._cluster(papers),
            self._summarize(papers),
        )
        papers = _apply_enrichment(papers, entity_lists, summaries, clusters)
        await self._remember(papers)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        response = SearchResponse(
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
            summaries=summary_report,
            cache=CacheReport(
                hit=False,
                ttl_seconds=self._search_cache.ttl_seconds if self._search_cache else None,
            ),
        )
        await self._cache(key, response)
        return response

    # --- caching ---------------------------------------------------------

    def _cache_key(self, request: SearchRequest) -> str | None:
        if self._search_cache is None:
            return None
        return cache_key(
            request.query,
            request.limit_per_source,
            request.sources,
            self._config_fingerprint(),
        )

    def _config_fingerprint(self) -> str:
        """Everything that changes what a search *returns*, not how fast.

        A response produced under a different ranking strategy or a
        different summary model is a different answer, so it must not be
        served after the setting changes — the same reason the summary
        cache keys on ``prompt_version``. Timeouts and log levels are
        deliberately absent: they change latency, not results.
        """
        settings = self._settings
        parts = [
            settings.ranking_strategy,
            f"{settings.ranking_alpha:.3f}",
            settings.embedding_model or "none",
            settings.summary_model if settings.mistral_api_key else "extractive",
            str(settings.entities_enabled),
        ]
        return "|".join(parts)

    async def _cached(self, key: str | None) -> tuple[SearchResponse, int] | None:
        """Look up a cached response, treating any failure as a miss."""
        if self._search_cache is None or key is None:
            return None
        try:
            return await asyncio.to_thread(self._search_cache.get, key)
        except Exception:
            logger.exception("search cache lookup failed; falling through to a live search")
            return None

    async def _cache(self, key: str | None, response: SearchResponse) -> None:
        """Store a response, but only a complete one.

        A degraded response must never be cached. If arXiv timed out,
        storing that for an hour turns one bad minute into a bad hour and
        hands every user in the window a two-source answer with no way to
        retry into a good one. A transient upstream failure should cost
        exactly as long as it lasts.
        """
        if self._search_cache is None or key is None:
            return
        if response.degraded:
            logger.debug("not caching a degraded response")
            return
        try:
            await asyncio.to_thread(self._search_cache.put, key, response)
        except Exception:
            logger.exception("could not cache the search response")

    async def _remember(self, papers: list[Paper]) -> None:
        """Persist the fully enriched papers so they can be exported later.

        Deliberately after enrichment: an exported citation should carry
        the merged record the user actually saw, not the raw retrieval
        output. A storage failure is logged and swallowed - losing the
        ability to export later must not cost the user their results now.
        """
        if self._paper_store is None or not papers:
            return
        try:
            await asyncio.to_thread(self._paper_store.save_many, papers)
        except Exception:
            logger.exception("could not persist papers for export")

    async def _extract_entities(
        self, papers: list[Paper]
    ) -> tuple[list[list[Entity]], EnrichmentReport]:
        """Find named entities per paper, degrading to none on failure."""
        if self._entity_extractor is None or not papers:
            return [], EnrichmentReport(
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
            return [], EnrichmentReport(applied=False, reason=f"NER failed: {exc}")

        return per_paper, EnrichmentReport(
            applied=True,
            model=self._entity_extractor.model_id,
            entities_found=sum(len(found) for found in per_paper),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _summarize(
        self, papers: list[Paper]
    ) -> tuple[list[Summary], SummaryReport]:
        """Summarize each paper, degrading to none on failure.

        The summarizer already falls back to extractive text internally, so
        reaching the except branch means something outside that policy
        broke — and a search is still worth returning without summaries.
        """
        if self._summarizer is None or not papers:
            return [], SummaryReport(
                applied=False,
                reason="summarization is disabled"
                if self._summarizer is None
                else "no papers to summarize",
            )
        try:
            summarized, report = await self._summarizer.summarize(papers)
        except Exception as exc:
            logger.exception("summarization failed")
            return [], SummaryReport(applied=False, reason=f"summarization failed: {exc}")
        return [paper.summary for paper in summarized if paper.summary], report

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


def _apply_enrichment(
    papers: list[Paper],
    entity_lists: Sequence[Sequence[Entity]],
    summaries: Sequence[Summary],
    clusters: Sequence[TopicCluster],
) -> list[Paper]:
    """Fold every enrichment pass onto the papers in one pass.

    Each pass produced its own output independently and possibly not at
    all, so this is where partial results get reconciled: a stage that
    returned nothing simply contributes nothing, and the others still land.

    Cluster membership is denormalized onto the paper as well as listed on
    the cluster because the results list renders per-paper and should not
    have to search every cluster's membership to find a badge.
    """
    cluster_of = {
        paper_id: cluster.id for cluster in clusters for paper_id in cluster.paper_ids
    }
    enriched: list[Paper] = []
    for index, paper in enumerate(papers):
        updates: dict[str, object] = {}
        if index < len(entity_lists):
            updates["entities"] = list(entity_lists[index])
        if index < len(summaries):
            updates["summary"] = summaries[index]
        if paper.id in cluster_of:
            updates["cluster_id"] = cluster_of[paper.id]
        enriched.append(paper.model_copy(update=updates) if updates else paper)
    return enriched


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
