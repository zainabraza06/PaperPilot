"""Exception hierarchy for the retrieval layer.

Connectors raise these instead of leaking ``httpx`` exceptions, so the
search service can map every failure mode onto a user-facing
``SourceStatus`` without importing the HTTP client.
"""

from __future__ import annotations


class PaperPilotError(Exception):
    """Base class for all application errors."""


class SourceError(PaperPilotError):
    """A source failed to answer a request."""

    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.message = message


class SourceUnavailableError(SourceError):
    """The upstream API returned a server error or could not be reached."""


class SourceRateLimitedError(SourceError):
    """The upstream API rejected us for sending too many requests."""

    def __init__(self, source: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(source, message)
        self.retry_after = retry_after


class SourceTimeoutError(SourceError):
    """The upstream API did not answer within the configured budget."""


class SourceParseError(SourceError):
    """The upstream API answered, but with a payload we could not parse."""
