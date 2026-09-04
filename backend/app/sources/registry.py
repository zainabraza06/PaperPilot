"""Source registry.

One place that knows which connectors exist. Adding a provider means
adding a class in this package and one line here — nothing else in the
codebase names the concrete connectors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import httpx

from app.config import Settings
from app.models.paper import SourceName
from app.sources.arxiv import ArxivSource
from app.sources.base import PaperSource
from app.sources.crossref import CrossrefSource
from app.sources.pubmed import PubMedSource

_SOURCE_CLASSES = (PubMedSource, ArxivSource, CrossrefSource)


class SourceRegistry:
    """Owns the connector instances and the HTTP client they share.

    A single ``AsyncClient`` is shared across connectors so that connection
    pooling and DNS caching apply across the whole fan-out; each connector
    keeps its own rate limiter because the politeness policies differ.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )
        self._sources: dict[SourceName, PaperSource] = {
            cls.name: cls(settings, self._client) for cls in _SOURCE_CLASSES
        }

    def all(self) -> Sequence[PaperSource]:
        """Every registered connector, in a stable order."""
        return list(self._sources.values())

    def get(self, name: SourceName) -> PaperSource:
        return self._sources[name]

    def select(self, names: Iterable[SourceName] | None) -> Sequence[PaperSource]:
        """Resolve a requested subset of sources, defaulting to all of them."""
        if not names:
            return self.all()
        return [self._sources[name] for name in names if name in self._sources]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
