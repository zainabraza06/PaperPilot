"""FastAPI dependencies.

Long-lived collaborators (the HTTP client, the connectors, the search
service) are built once during application startup and handed to routes
from here, so routes never construct infrastructure themselves.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.services.search_service import SearchService
from app.sources.registry import SourceRegistry
from app.storage.paper_store import PaperStore


def get_registry(request: Request) -> SourceRegistry:
    """Read the registry built during application startup."""
    # Starlette's State is an untyped attribute bag by design.
    return cast(SourceRegistry, request.app.state.registry)


def get_search_service(request: Request) -> SearchService:
    """Read the search service built during application startup."""
    return cast(SearchService, request.app.state.search_service)


def get_paper_store(request: Request) -> PaperStore:
    """Read the paper store, refusing clearly if storage is switched off.

    Export cannot answer at all without it, so an absent store is a
    configuration error worth naming rather than something to degrade past.
    """
    store = getattr(request.app.state, "paper_store", None)
    if store is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "paper storage is disabled; set PAPERPILOT_PAPER_STORE_PATH to enable export",
        )
    return cast(PaperStore, store)


SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[SourceRegistry, Depends(get_registry)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
PaperStoreDep = Annotated[PaperStore, Depends(get_paper_store)]
