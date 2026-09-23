"""Transport-level behaviour of the connectors.

Parsing is covered by the per-source tests; this file covers the parts that
only show up against a live-ish HTTP layer: retries, error translation,
politeness parameters, and the two-call PubMed flow.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.core.errors import SourceRateLimitedError, SourceUnavailableError
from app.models.search import SourceQuery
from app.services.query_parser import parse_query
from app.sources.arxiv import ArxivSource
from app.sources.crossref import CrossrefSource
from app.sources.pubmed import PubMedSource
from tests.conftest import load_fixture


@pytest.fixture
def fast_settings(settings: Settings) -> Settings:
    """Retries enabled, but with no real sleeping between attempts."""
    return settings.model_copy(update={"http_max_retries": 2, "http_timeout_seconds": 1.0})


def query(text: str = "prime editing", limit: int = 3) -> SourceQuery:
    return SourceQuery(parsed=parse_query(text), limit=limit)


@respx.mock
async def test_pubmed_makes_the_two_calls_and_preserves_relevance_order(
    fast_settings: Settings,
) -> None:
    esearch = respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_esearch.json"))
    )
    efetch = respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_efetch.xml"))
    )

    source = PubMedSource(fast_settings)
    papers = await source.search(query())
    await source.aclose()

    assert esearch.called and efetch.called
    assert [p.source_id for p in papers] == ["35123456", "31987654", "28001122"]
    # The PMIDs esearch chose must all be requested in one efetch call.
    efetch_params = efetch.calls[0].request.url.params
    assert efetch_params["id"] == "35123456,31987654,28001122"


@respx.mock
async def test_pubmed_skips_efetch_when_esearch_finds_nothing(fast_settings: Settings) -> None:
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch").mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}})
    )
    efetch = respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch")

    source = PubMedSource(fast_settings)
    assert await source.search(query()) == []
    await source.aclose()
    assert not efetch.called


@respx.mock
async def test_crossref_sends_a_mailto_for_the_polite_pool(fast_settings: Settings) -> None:
    route = respx.get(url__startswith="https://api.crossref.org/works").mock(
        return_value=httpx.Response(200, text=load_fixture("crossref_works.json"))
    )

    source = CrossrefSource(fast_settings)
    papers = await source.search(query())
    await source.aclose()

    assert len(papers) == 3
    assert "mailto=tests%40example.com" in str(route.calls[0].request.url)


@respx.mock
async def test_crossref_unknown_doi_is_not_found_rather_than_an_error(
    fast_settings: Settings,
) -> None:
    respx.get(url__startswith="https://api.crossref.org/works/").mock(
        return_value=httpx.Response(404, text="Resource not found.")
    )

    source = CrossrefSource(fast_settings)
    assert await source.fetch_by_doi("10.1234/does-not-exist") is None
    await source.aclose()


@respx.mock
async def test_arxiv_search_parses_the_atom_feed(fast_settings: Settings) -> None:
    respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=load_fixture("arxiv_feed.xml"))
    )

    source = ArxivSource(fast_settings)
    papers = await source.search(query())
    await source.aclose()

    assert [p.source_id for p in papers] == ["1706.03762", "2101.00001"]


@respx.mock
async def test_transient_server_errors_are_retried(fast_settings: Settings) -> None:
    route = respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, text=load_fixture("arxiv_feed.xml")),
        ]
    )

    source = ArxivSource(fast_settings)
    papers = await source.search(query())
    await source.aclose()

    assert route.call_count == 2
    assert len(papers) == 2


@respx.mock
async def test_retries_are_bounded_and_then_reported(fast_settings: Settings) -> None:
    route = respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(500)
    )

    source = ArxivSource(fast_settings)
    with pytest.raises(SourceUnavailableError):
        await source.search(query())
    await source.aclose()

    assert route.call_count == fast_settings.http_max_retries + 1


@respx.mock
async def test_client_errors_are_not_retried(fast_settings: Settings) -> None:
    # A 400 means the request itself is wrong; retrying only wastes quota.
    route = respx.get(url__startswith="https://api.crossref.org/works").mock(
        return_value=httpx.Response(400)
    )

    source = CrossrefSource(fast_settings)
    with pytest.raises(SourceUnavailableError, match="HTTP 400"):
        await source.search(query())
    await source.aclose()

    assert route.call_count == 1


@respx.mock
async def test_rate_limiting_is_reported_distinctly(fast_settings: Settings) -> None:
    respx.get(url__startswith="https://api.crossref.org/works").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"})
    )

    source = CrossrefSource(fast_settings)
    with pytest.raises(SourceRateLimitedError):
        await source.search(query())
    await source.aclose()


@respx.mock
async def test_timeouts_are_translated_not_leaked(fast_settings: Settings) -> None:
    from app.core.errors import SourceTimeoutError

    respx.get(url__startswith="https://api.crossref.org/works").mock(
        side_effect=httpx.ReadTimeout("too slow")
    )

    source = CrossrefSource(fast_settings)
    with pytest.raises(SourceTimeoutError):
        await source.search(query())
    await source.aclose()


@respx.mock
async def test_arxiv_resolves_its_own_datacite_doi_without_searching(
    fast_settings: Settings,
) -> None:
    """An arXiv DOI carries the arXiv id; it should not be text-searched.

    Since 2022 arXiv mints 10.48550/arXiv.<id> for every submission.
    Crossref and PubMed legitimately hold no DataCite records, so if arXiv
    also fails to resolve it, a perfectly valid DOI returns nothing at all
    from any source - which is exactly what happened before this.
    """
    route = respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=load_fixture("arxiv_feed.xml"))
    )

    source = ArxivSource(fast_settings)
    paper = await source.fetch_by_doi("10.48550/arXiv.1706.03762")
    await source.aclose()

    assert paper is not None
    # Resolved by id_list, not by hunting for the DOI string in full text.
    request_url = str(route.calls[0].request.url)
    assert "id_list=1706.03762" in request_url
    assert "search_query" not in request_url


@respx.mock
async def test_a_foreign_doi_still_falls_back_to_a_text_hunt(
    fast_settings: Settings,
) -> None:
    route = respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=load_fixture("arxiv_feed.xml"))
    )

    source = ArxivSource(fast_settings)
    await source.fetch_by_doi("10.1038/s41587-022-01234-5")
    await source.aclose()

    assert "search_query" in str(route.calls[0].request.url)
