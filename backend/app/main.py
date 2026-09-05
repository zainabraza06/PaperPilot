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
from app.services.enrichment.clustering import TopicClusterer
from app.services.enrichment.entities import EntityExtractor, build_entity_extractor
from app.services.ranking.cache import CachedEmbedder
from app.services.ranking.embeddings import build_embedder
from app.services.ranking.hybrid import FusionStrategy, HybridRanker
from app.services.search_service import SearchService
from app.sources.registry import SourceRegistry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    registry = SourceRegistry(settings)
    embedder = await _build_embedder(settings)
    ranker = _build_ranker(settings, embedder)
    clusterer = (
        TopicClusterer(
            embedder,
            max_clusters=settings.max_clusters,
            min_papers=settings.min_papers_to_cluster,
        )
        if embedder is not None and settings.clustering_enabled
        else None
    )
    extractor = await _build_entity_extractor(settings)

    app.state.registry = registry
    app.state.embedder = embedder
    app.state.ranker = ranker
    app.state.clusterer = clusterer
    app.state.entity_extractor = extractor
    app.state.search_service = SearchService(registry, settings, ranker, extractor, clusterer)
    logger.info(
        "%s started | sources: %s | ranking: %s | entities: %s | clustering: %s",
        settings.app_name,
        ", ".join(source.display_name for source in registry.all()),
        f"{ranker.strategy.value} on {ranker.model_id}" if ranker else "disabled",
        extractor.model_id if extractor else "disabled",
        "on" if clusterer else "disabled",
    )
    try:
        yield
    finally:
        await registry.aclose()
        logger.info("shutdown complete")


async def _build_embedder(settings: Settings) -> CachedEmbedder | None:
    """Load the embedding model once, wrapped in a shared cache.

    One instance is shared by the ranker and the clusterer, which both embed
    the same papers during a single search: the second consumer gets them
    from the cache instead of paying for another forward pass.

    Loading is synchronous and slow, so it runs in a thread. If the model
    cannot be loaded at all, ``build_embedder`` degrades to the hashing
    fallback rather than leaving the service unable to rank.
    """
    if not (settings.ranking_enabled or settings.clustering_enabled):
        return None
    return CachedEmbedder(
        await asyncio.to_thread(build_embedder, settings.embedding_model)
    )


def _build_ranker(
    settings: Settings, embedder: CachedEmbedder | None
) -> HybridRanker | None:
    """Construct the ranker, or ``None`` if ranking is switched off."""
    if not settings.ranking_enabled or embedder is None:
        return None
    return HybridRanker(
        embedder,
        strategy=FusionStrategy(settings.ranking_strategy),
        alpha=settings.ranking_alpha,
    )


async def _build_entity_extractor(settings: Settings) -> EntityExtractor | None:
    """Load the NER pipeline, or ``None`` if entity extraction is off."""
    if not settings.entities_enabled:
        return None
    return await asyncio.to_thread(build_entity_extractor, settings.ner_model)


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
