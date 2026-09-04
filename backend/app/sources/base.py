"""The ``PaperSource`` interface every connector implements.

The contract is deliberately narrow — search, and look up by identifier —
so that adding a fourth provider (Semantic Scholar, OpenAlex, …) is a new
file in this package and one registry entry, with no changes anywhere else.

``BaseHttpSource`` supplies the parts that are genuinely shared: an httpx
client, politeness rate limiting, and retry-with-backoff that translates
transport failures into the ``SourceError`` hierarchy.
"""

from __future__ import annotations

import abc
import asyncio
import random
from typing import Any

import httpx

from app.config import Settings
from app.core.errors import (
    SourceParseError,
    SourceRateLimitedError,
    SourceTimeoutError,
    SourceUnavailableError,
)
from app.core.logging import get_logger
from app.core.ratelimit import AsyncRateLimiter
from app.models.paper import Paper, SourceName
from app.models.search import SourceQuery

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class PaperSource(abc.ABC):
    """A provider of scientific papers.

    Implementations must be safe to share across concurrent requests and
    must never raise transport-level exceptions; failures are reported as
    ``SourceError`` subclasses so the search service can degrade cleanly.
    """

    #: Provider identity, used for registry lookup and result attribution.
    name: SourceName

    #: Human-readable label for UI badges and log lines.
    display_name: str

    @abc.abstractmethod
    async def search(self, query: SourceQuery) -> list[Paper]:
        """Return up to ``query.limit`` papers matching ``query``."""

    @abc.abstractmethod
    async def fetch_by_doi(self, doi: str) -> Paper | None:
        """Return the single paper with this DOI, or ``None`` if unknown."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not every source holds resources
        """Release any held resources."""


class BaseHttpSource(PaperSource):
    """Shared HTTP plumbing for connectors that talk to a REST/XML endpoint."""

    #: Minimum seconds between outbound requests, per the provider's policy.
    min_request_interval: float = 0.0

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )
        self._limiter = AsyncRateLimiter(self._rate_limit_interval())
        self._log = get_logger(f"paperpilot.sources.{self.name.value}")

    def _rate_limit_interval(self) -> float:
        """Seconds to leave between outbound requests.

        Overridden by connectors whose politeness policy depends on runtime
        configuration (for example, PubMed's allowance doubles with a key).
        """
        return self.min_request_interval

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """GET with politeness limiting, bounded retries, and typed errors.

        Retries only on 429/5xx and transport errors — a 400 from a
        malformed query will never succeed on a second attempt.
        """
        attempts = self.settings.http_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.get(url, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                last_error = SourceTimeoutError(self.name.value, f"request timed out: {exc}")
            except httpx.HTTPError as exc:
                last_error = SourceUnavailableError(self.name.value, f"transport error: {exc}")
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in _RETRYABLE_STATUS:
                    raise SourceUnavailableError(
                        self.name.value,
                        f"HTTP {response.status_code} for {response.request.url}",
                    )
                retry_after = _parse_retry_after(response)
                last_error = (
                    SourceRateLimitedError(
                        self.name.value, "upstream rate limit hit", retry_after
                    )
                    if response.status_code == 429
                    else SourceUnavailableError(
                        self.name.value, f"HTTP {response.status_code} from upstream"
                    )
                )
                if attempt < attempts:
                    await asyncio.sleep(retry_after or self._backoff(attempt))
                    continue

            if attempt < attempts:
                self._log.warning(
                    "attempt %d/%d failed (%s); retrying", attempt, attempts, last_error
                )
                await asyncio.sleep(self._backoff(attempt))

        assert last_error is not None  # loop always records a failure before exhausting
        raise last_error

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with jitter, capped so a search stays snappy."""
        return min(2.0 ** (attempt - 1), 4.0) + random.uniform(0, 0.25)

    def _parse_error(self, message: str) -> SourceParseError:
        return SourceParseError(self.name.value, message)


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Read a ``Retry-After`` header, tolerating absence and garbage."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 10.0))
    except ValueError:
        return None
