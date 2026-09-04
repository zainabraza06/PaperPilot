"""Shared test fixtures.

The connector tests are deliberately offline: they exercise the parsing
and normalization logic against recorded payloads, which is the part that
breaks when an upstream schema shifts. Network behaviour (retries, status
mapping) is tested separately with mocked transports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.sources.arxiv import ArxivSource
from app.sources.crossref import CrossrefSource
from app.sources.pubmed import PubMedSource

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    """Read a recorded upstream payload."""
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    """Settings with rate limiting disabled so tests do not sleep."""
    return Settings(
        arxiv_min_interval_seconds=0.0,
        http_max_retries=1,
        source_timeout_seconds=2.0,
        crossref_mailto="tests@example.com",
    )


@pytest.fixture
def pubmed(settings: Settings) -> PubMedSource:
    return PubMedSource(settings)


@pytest.fixture
def arxiv(settings: Settings) -> ArxivSource:
    return ArxivSource(settings)


@pytest.fixture
def crossref(settings: Settings) -> CrossrefSource:
    return CrossrefSource(settings)
