"""Search orchestration tests.

The contract under test: *one flaky source must never break a search*.
Every failure mode is expected to surface as a per-source status alongside
whatever results the healthy sources returned.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.core.errors import (
    SourceParseError,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from app.models.paper import Paper, SourceName
from app.models.search import SearchRequest, SourceQuery, SourceStatus
from app.services.ranking.embeddings import HashingEmbedder
from app.services.ranking.hybrid import HybridRanker
from app.services.search_service import SearchService
from app.sources.base import PaperSource


class FakeSource(PaperSource):
    """A connector stand-in with scriptable behaviour."""

    def __init__(
        self,
        name: SourceName,
        papers: list[Paper] | None = None,
        *,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self.display_name = name.value.title()
        self._papers = papers or []
        self._error = error
        self._delay = delay
        self.search_calls: list[SourceQuery] = []
        self.doi_lookups: list[str] = []

    async def search(self, query: SourceQuery) -> list[Paper]:
        self.search_calls.append(query)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return list(self._papers)

    async def fetch_by_doi(self, doi: str) -> Paper | None:
        self.doi_lookups.append(doi)
        if self._error:
            raise self._error
        return self._papers[0] if self._papers else None


class FakeRegistry:
    def __init__(self, sources: list[PaperSource]) -> None:
        self._sources = sources

    def all(self) -> list[PaperSource]:
        return self._sources

    def select(self, names: list[SourceName] | None) -> list[PaperSource]:
        if not names:
            return self._sources
        return [s for s in self._sources if s.name in names]


def make_paper(source: SourceName, n: int, doi: str | None = None) -> Paper:
    return Paper(
        id=f"{source.value}:{n}",
        doi=doi,
        source=source,
        source_id=str(n),
        title=f"{source.value} paper {n}",
        url=f"https://example.org/{source.value}/{n}",
    )


def build_service(
    sources: list[PaperSource], settings: Settings, ranker: HybridRanker | None = None
) -> SearchService:
    return SearchService(FakeRegistry(sources), settings, ranker)  # type: ignore[arg-type]


async def test_all_sources_are_queried_concurrently(settings: Settings) -> None:
    sources = [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1)], delay=0.15),
        FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)], delay=0.15),
        FakeSource(SourceName.CROSSREF, [make_paper(SourceName.CROSSREF, 1)], delay=0.15),
    ]
    service = build_service(sources, settings)

    loop = asyncio.get_running_loop()
    started = loop.time()
    response = await service.search(SearchRequest(query="prime editing"))
    elapsed = loop.time() - started

    assert response.total == 3
    # Sequential execution would take ~0.45s; concurrent is bounded by one delay.
    assert elapsed < 0.4


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (SourceUnavailableError("pubmed", "HTTP 503"), SourceStatus.UNAVAILABLE),
        (SourceRateLimitedError("pubmed", "429"), SourceStatus.RATE_LIMITED),
        (SourceParseError("pubmed", "bad xml"), SourceStatus.PARSE_ERROR),
        (RuntimeError("connector bug"), SourceStatus.UNAVAILABLE),
    ],
)
async def test_a_failing_source_degrades_instead_of_breaking_the_search(
    settings: Settings, error: Exception, expected: SourceStatus
) -> None:
    sources = [
        FakeSource(SourceName.PUBMED, error=error),
        FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)]),
        FakeSource(SourceName.CROSSREF, [make_paper(SourceName.CROSSREF, 1)]),
    ]
    response = await build_service(sources, settings).search(SearchRequest(query="crispr"))

    assert response.total == 2
    assert response.degraded is True
    reports = {r.source: r for r in response.sources}
    assert reports[SourceName.PUBMED].status is expected
    assert reports[SourceName.PUBMED].message
    assert reports[SourceName.ARXIV].status is SourceStatus.OK


async def test_a_slow_source_is_cut_off_at_its_timeout(settings: Settings) -> None:
    sources = [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1)], delay=10.0),
        FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)]),
    ]
    service = build_service(sources, settings.model_copy(update={"source_timeout_seconds": 0.05}))
    response = await service.search(SearchRequest(query="crispr"))

    reports = {r.source: r for r in response.sources}
    assert reports[SourceName.PUBMED].status is SourceStatus.TIMEOUT
    assert response.total == 1


async def test_a_source_with_no_matches_reports_empty_not_failure(settings: Settings) -> None:
    sources = [
        FakeSource(SourceName.PUBMED, []),
        FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)]),
    ]
    response = await build_service(sources, settings).search(SearchRequest(query="crispr"))

    reports = {r.source: r for r in response.sources}
    assert reports[SourceName.PUBMED].status is SourceStatus.EMPTY
    # "PubMed found nothing" is an informative state, not a degraded search.
    assert response.degraded is False


async def test_every_source_failing_yields_an_empty_but_well_formed_response(
    settings: Settings,
) -> None:
    sources = [
        FakeSource(name, error=SourceUnavailableError(name.value, "HTTP 500"))
        for name in SourceName
    ]
    response = await build_service(sources, settings).search(SearchRequest(query="crispr"))

    assert response.papers == []
    assert response.total == 0
    assert len(response.sources) == 3
    assert all(r.status is SourceStatus.UNAVAILABLE for r in response.sources)


async def test_duplicates_across_sources_are_merged_and_counted(settings: Settings) -> None:
    doi = "10.1234/shared"
    sources = [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1, doi)]),
        FakeSource(SourceName.CROSSREF, [make_paper(SourceName.CROSSREF, 1, doi)]),
    ]
    response = await build_service(sources, settings).search(SearchRequest(query="crispr"))

    assert response.total == 1
    assert response.duplicates_merged == 1
    assert response.papers[0].also_found_in == [SourceName.CROSSREF]


async def test_results_are_interleaved_so_no_single_source_leads(settings: Settings) -> None:
    sources = [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, n) for n in range(3)]),
        FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, n) for n in range(3)]),
    ]
    response = await build_service(sources, settings).search(SearchRequest(query="crispr"))

    assert [p.source for p in response.papers[:4]] == [
        SourceName.PUBMED,
        SourceName.ARXIV,
        SourceName.PUBMED,
        SourceName.ARXIV,
    ]


async def test_a_doi_query_triggers_a_lookup_rather_than_a_search(settings: Settings) -> None:
    sources = [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1, "10.1234/abc")]),
        FakeSource(SourceName.CROSSREF, [make_paper(SourceName.CROSSREF, 1, "10.1234/abc")]),
    ]
    response = await build_service(sources, settings).search(SearchRequest(query="10.1234/abc"))

    assert all(source.search_calls == [] for source in sources)
    assert all(source.doi_lookups == ["10.1234/abc"] for source in sources)
    assert response.total == 1


async def test_sources_can_be_restricted_per_request(settings: Settings) -> None:
    sources = [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1)]),
        FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)]),
    ]
    service = build_service(sources, settings)
    response = await service.search(
        SearchRequest(query="crispr", sources=[SourceName.ARXIV])
    )

    assert [r.source for r in response.sources] == [SourceName.ARXIV]
    assert response.total == 1


async def test_the_per_source_limit_is_passed_through(settings: Settings) -> None:
    source = FakeSource(SourceName.ARXIV, [])
    await build_service([source], settings).search(
        SearchRequest(query="crispr", limit_per_source=7)
    )
    assert source.search_calls[0].limit == 7


# --- ranking integration -------------------------------------------------


def make_titled_paper(source: SourceName, n: int, title: str, abstract: str) -> Paper:
    return Paper(
        id=f"{source.value}:{n}",
        source=source,
        source_id=str(n),
        title=title,
        abstract=abstract,
        url=f"https://example.org/{n}",
    )


async def test_results_are_reordered_by_relevance_when_a_ranker_is_present(
    settings: Settings,
) -> None:
    papers = [
        make_titled_paper(SourceName.ARXIV, 1, "Solar cell efficiency", "photovoltaic silicon"),
        make_titled_paper(SourceName.ARXIV, 2, "Antenna arrays", "beamforming design"),
        make_titled_paper(SourceName.ARXIV, 3, "Ocean acidification", "coral reef surveys"),
        make_titled_paper(
            SourceName.ARXIV, 4, "Prime editing efficiency", "prime editing in human cells"
        ),
    ]
    service = build_service(
        [FakeSource(SourceName.ARXIV, papers)], settings, HybridRanker(HashingEmbedder())
    )
    response = await service.search(SearchRequest(query="prime editing"))

    assert response.papers[0].title == "Prime editing efficiency"
    assert response.ranking.applied is True
    assert response.ranking.strategy == "linear"
    assert response.ranking.model.startswith("hashing-")


async def test_every_returned_paper_carries_a_score(settings: Settings) -> None:
    papers = [make_paper(SourceName.ARXIV, n) for n in range(3)]
    service = build_service(
        [FakeSource(SourceName.ARXIV, papers)], settings, HybridRanker(HashingEmbedder())
    )
    response = await service.search(SearchRequest(query="arxiv paper"))

    assert all(paper.score is not None for paper in response.papers)
    assert [paper.score.rank for paper in response.papers] == [1, 2, 3]


async def test_the_ranker_sees_the_raw_query_not_the_distilled_keywords(
    settings: Settings,
) -> None:
    """The reason ParsedQuery keeps ``raw`` at all.

    Upstream APIs get a keyword string because none of them accepts a
    paragraph; the ranker gets the whole pasted abstract, which is what
    makes semantic matching on a snippet work.
    """
    seen: list[str] = []

    class RecordingRanker(HybridRanker):
        async def arank(self, query: str, papers):  # type: ignore[no-untyped-def]
            seen.append(query)
            return list(papers)

    abstract = (
        "We present a method for correcting technical variation between batches of "
        "single-cell transcriptomic data using mutual nearest neighbours to estimate "
        "a correction vector without requiring shared populations across batches."
    )
    service = build_service(
        [FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)])],
        settings,
        RecordingRanker(HashingEmbedder()),
    )
    response = await service.search(SearchRequest(query=abstract))

    assert seen == [abstract]
    assert response.query.search_terms != abstract
    assert len(response.query.search_terms.split()) <= 10


async def test_a_ranking_failure_degrades_to_retrieval_order(settings: Settings) -> None:
    class BrokenRanker(HybridRanker):
        async def arank(self, query: str, papers):  # type: ignore[no-untyped-def]
            raise RuntimeError("model exploded")

    papers = [make_paper(SourceName.ARXIV, n) for n in range(3)]
    service = build_service(
        [FakeSource(SourceName.ARXIV, papers)], settings, BrokenRanker(HashingEmbedder())
    )
    response = await service.search(SearchRequest(query="crispr"))

    # An unranked list of the right papers still beats no results at all.
    assert response.total == 3
    assert response.ranking.applied is False
    assert "model exploded" in response.ranking.reason


async def test_ranking_disabled_is_reported_rather_than_hidden(settings: Settings) -> None:
    service = build_service([FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)])], settings)
    response = await service.search(SearchRequest(query="crispr"))

    assert response.ranking.applied is False
    assert response.ranking.reason == "ranking is disabled"
    assert response.papers[0].score is None
