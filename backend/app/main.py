"""FastAPI application factory and process lifecycle.

The connector registry and search service are created once at startup and
torn down at shutdown, so the shared HTTP connection pool lives for the
lifetime of the process rather than per request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, search
from app.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.services.search_service import SearchService
from app.sources.registry import SourceRegistry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    registry = SourceRegistry(settings)
    app.state.registry = registry
    app.state.search_service = SearchService(registry, settings)
    logger.info(
        "%s started with sources: %s",
        settings.app_name,
        ", ".join(source.display_name for source in registry.all()),
    )
    try:
        yield
    finally:
        await registry.aclose()
        logger.info("shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings for tests."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="AI-powered scientific literature search across PubMed, arXiv and Crossref.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(search.router)
    return app


app = create_app()
