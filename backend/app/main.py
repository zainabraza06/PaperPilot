"""FastAPI application factory and process lifecycle.

The connector registry, the embedding model and the search service are
created once at startup and torn down at shutdown, so the shared HTTP
connection pool lives for the lifetime of the process rather than per
request.

Loading the embedding model costs a few seconds of startup time. That is
paid deliberately at boot rather than on a user's first search, and it is
done in a worker thread so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, search
from app.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.services.ranking.embeddings import build_embedder
from app.services.ranking.hybrid import FusionStrategy, HybridRanker
from app.services.search_service import SearchService
from app.sources.registry import SourceRegistry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    registry = SourceRegistry(settings)
    ranker = await _build_ranker(settings)
    app.state.registry = registry
    app.state.ranker = ranker
    app.state.search_service = SearchService(registry, settings, ranker)
    logger.info(
        "%s started with sources: %s | ranking: %s",
        settings.app_name,
        ", ".join(source.display_name for source in registry.all()),
        f"{ranker.strategy.value} on {ranker.model_id}" if ranker else "disabled",
    )
    try:
        yield
    finally:
        await registry.aclose()
        logger.info("shutdown complete")


async def _build_ranker(settings: Settings) -> HybridRanker | None:
    """Construct the ranker, or ``None`` if ranking is switched off.

    Model loading is synchronous and slow, so it runs in a thread. If the
    model cannot be loaded at all, ``build_embedder`` degrades to the
    hashing fallback rather than leaving the service unable to rank.
    """
    if not settings.ranking_enabled:
        return None
    embedder = await asyncio.to_thread(build_embedder, settings.embedding_model)
    return HybridRanker(
        embedder,
        strategy=FusionStrategy(settings.ranking_strategy),
        alpha=settings.ranking_alpha,
    )


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
