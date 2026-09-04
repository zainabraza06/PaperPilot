"""API surface tests.

Routes are thin, so these check wiring and contract shape rather than
behaviour: the service layer is where the logic tests live.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_search_service
from app.config import Settings
from app.main import create_app
from app.models.paper import Paper, SourceName
from app.models.search import (
    SearchRequest,
    SearchResponse,
    SourceReport,
    SourceStatus,
)
from app.services.query_parser import parse_query


class StubSearchService:
    """Returns a fixed, degraded response so the shape can be asserted."""

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        parsed = parse_query(request.query)
        paper = Paper(
            id="doi:10.1234/abc",
            doi="10.1234/abc",
            source=SourceName.ARXIV,
            source_id="2101.00001",
            title="A paper",
            url="https://arxiv.org/abs/2101.00001",
            also_found_in=[SourceName.CROSSREF],
        )
        return SearchResponse(
            query=parsed,
            papers=[paper],
            total=1,
            sources=[
                SourceReport(source=SourceName.PUBMED, status=SourceStatus.UNAVAILABLE,
                             message="PubMed is unreachable"),
                SourceReport(source=SourceName.ARXIV, status=SourceStatus.OK, returned=1),
                SourceReport(source=SourceName.CROSSREF, status=SourceStatus.EMPTY),
            ],
            elapsed_ms=42,
            duplicates_merged=1,
        )


@pytest.fixture
def stub() -> StubSearchService:
    return StubSearchService()


@pytest.fixture
def client(stub: StubSearchService) -> Iterator[TestClient]:
    app = create_app(Settings(environment="test"))
    app.dependency_overrides[get_search_service] = lambda: stub
    with TestClient(app) as test_client:
        yield test_client


def test_health_lists_the_registered_sources(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert {s["name"] for s in body["sources"]} == {"pubmed", "arxiv", "crossref"}


def test_search_returns_papers_and_per_source_reports(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "prime editing"}).json()

    assert body["total"] == 1
    assert body["duplicates_merged"] == 1
    assert body["papers"][0]["also_found_in"] == ["crossref"]
    # The partial-failure detail the UI needs to explain itself.
    statuses = {s["source"]: s["status"] for s in body["sources"]}
    assert statuses == {"pubmed": "unavailable", "arxiv": "ok", "crossref": "empty"}


def test_search_passes_through_limit_and_source_filters(
    client: TestClient, stub: StubSearchService
) -> None:
    client.get(
        "/api/search",
        params={"q": "crispr", "limit_per_source": 5, "sources": ["arxiv", "crossref"]},
    )
    request = stub.requests[-1]
    assert request.limit_per_source == 5
    assert request.sources == [SourceName.ARXIV, SourceName.CROSSREF]


def test_post_accepts_a_long_abstract_body(client: TestClient, stub: StubSearchService) -> None:
    abstract = "Prime editing enables precise genome edits. " * 20
    response = client.post("/api/search", json={"query": abstract})
    assert response.status_code == 200
    assert stub.requests[-1].query == abstract


def test_parse_endpoint_exposes_the_detected_intent(client: TestClient) -> None:
    body = client.get("/api/parse", params={"q": "10.1038/s41587-022-01234-5"}).json()
    assert body["intent"] == "identifier"
    assert body["identifier_kind"] == "doi"


def test_empty_query_is_rejected_with_a_422(client: TestClient) -> None:
    assert client.get("/api/search", params={"q": ""}).status_code == 422


def test_out_of_range_limit_is_rejected(client: TestClient) -> None:
    assert client.get("/api/search", params={"q": "x", "limit_per_source": 500}).status_code == 422
