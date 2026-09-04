"""Search endpoints.

Routes stay thin on purpose: validate, delegate to ``SearchService``,
return. All orchestration, error handling and merging lives in the service
layer, which is what the tests exercise.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import SearchServiceDep
from app.models.paper import SourceName
from app.models.search import ParsedQuery, SearchRequest, SearchResponse
from app.services.query_parser import parse_query

router = APIRouter(prefix="/api", tags=["search"])


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search PubMed, arXiv and Crossref in one query",
)
async def search(
    service: SearchServiceDep,
    q: str = Query(min_length=1, max_length=8000, description="Topic, keyword, DOI or abstract"),
    limit_per_source: int = Query(default=20, ge=1, le=100),
    sources: list[SourceName] | None = Query(default=None),
) -> SearchResponse:
    """Fan out to every source, merge, deduplicate, and report per-source status."""
    return await _run(service, SearchRequest(query=q, limit_per_source=limit_per_source, sources=sources))


@router.post("/search", response_model=SearchResponse, summary="Search with a JSON body")
async def search_post(request: SearchRequest, service: SearchServiceDep) -> SearchResponse:
    """POST variant, for abstract snippets too long to sit in a query string."""
    return await _run(service, request)


@router.get(
    "/parse",
    response_model=ParsedQuery,
    summary="Show how a query would be interpreted",
)
async def parse(q: str = Query(min_length=1, max_length=8000)) -> ParsedQuery:
    """Expose the query classifier so the UI can show the detected input type."""
    try:
        return parse_query(q)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


async def _run(service: SearchServiceDep, request: SearchRequest) -> SearchResponse:
    try:
        return await service.search(request)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
