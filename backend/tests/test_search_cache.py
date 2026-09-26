"""Search-response caching.

The fan-out is 3-7 seconds because it is bounded by the slowest upstream
API, and a repeated query re-pays all of it. These tests pin the two
things that make caching it safe rather than merely fast: that a cache hit
does not re-query the sources, and that a *degraded* response is never
stored.

That second one is the whole design. Caching a response where arXiv timed
out turns one bad minute into a bad hour, and hands every user in the
window a two-source answer with no way to retry into a good one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.core.errors import SourceUnavailableError
from app.models.paper import SourceName
from app.models.search import SearchRequest
from app.services.search_service import SearchService
from app.storage.search_cache import SearchCache, cache_key
from tests.test_search_service import FakeRegistry, FakeSource, make_paper

pytestmark = pytest.mark.asyncio


@pytest.fixture
def cache(tmp_path: Path) -> SearchCache:
    return SearchCache(tmp_path / "cache.db", ttl_seconds=3600)


def build_cached_service(
    sources: list[FakeSource], settings: Settings, cache: SearchCache | None
) -> SearchService:
    return SearchService(
        FakeRegistry(sources),  # type: ignore[arg-type]
        settings,
        search_cache=cache,
    )


def healthy_sources() -> list[FakeSource]:
    return [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1)]),
        FakeSource(SourceName.ARXIV, [make_paper(SourceName.ARXIV, 1)]),
    ]


# --- the happy path -----------------------------------------------------


async def test_a_repeated_search_does_not_touch_the_sources_again(
    settings: Settings, cache: SearchCache
) -> None:
    """The point of the feature, stated as the sources never being called."""
    sources = healthy_sources()
    service = build_cached_service(sources, settings, cache)
    request = SearchRequest(query="prime editing")

    first = await service.search(request)
    second = await service.search(request)

    assert first.cache.hit is False
    assert second.cache.hit is True
    assert all(len(source.search_calls) == 1 for source in sources)
    assert [p.id for p in second.papers] == [p.id for p in first.papers]


async def test_a_hit_reports_its_own_age_and_ttl(
    settings: Settings, cache: SearchCache
) -> None:
    """A silent cache would contradict the rest of the pipeline reporting."""
    service = build_cached_service(healthy_sources(), settings, cache)
    request = SearchRequest(query="prime editing")

    await service.search(request)
    hit = await service.search(request)

    assert hit.cache.hit is True
    assert hit.cache.age_seconds is not None and hit.cache.age_seconds >= 0
    assert hit.cache.ttl_seconds == 3600


async def test_elapsed_ms_describes_this_request_not_the_original(
    settings: Settings, cache: SearchCache
) -> None:
    """Replaying the original timing would make the cache invisible.

    The pipeline panel exists to show where the time went; a hit that
    still claims seven seconds would misreport exactly the thing the
    feature improves.
    """
    slow = [FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1)], delay=0.2)]
    service = build_cached_service(slow, settings, cache)
    request = SearchRequest(query="prime editing")

    first = await service.search(request)
    hit = await service.search(request)

    # Asserted against the *stored* payload rather than a wall-clock
    # threshold. An earlier version compared the two live timings and was
    # flaky under a loaded suite, where a SQLite read can take tens of
    # milliseconds — and a test that fails on scheduling noise teaches
    # people to re-run it rather than to read it.
    #
    # This is the same claim with no clock in it: the row on disk keeps
    # the original cost, and the response handed back does not.
    stored = cache.get(service._cache_key(request) or "")
    assert stored is not None
    assert stored[0].elapsed_ms == first.elapsed_ms >= 200
    assert hit.elapsed_ms != stored[0].elapsed_ms


# --- what must not be cached -------------------------------------------


async def test_a_degraded_response_is_never_cached(
    settings: Settings, cache: SearchCache
) -> None:
    """The rule the whole design rests on."""
    sources = [
        FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1)]),
        FakeSource(SourceName.ARXIV, error=SourceUnavailableError("arxiv", "HTTP 503")),
    ]
    service = build_cached_service(sources, settings, cache)
    request = SearchRequest(query="prime editing")

    first = await service.search(request)
    assert first.degraded is True
    assert cache.count() == 0

    # And the next search really does retry rather than replaying failure.
    second = await service.search(request)
    assert second.cache.hit is False
    assert len(sources[0].search_calls) == 2


async def test_a_source_that_recovers_is_cached_normally(
    settings: Settings, cache: SearchCache
) -> None:
    """A transient failure must cost exactly as long as it lasts."""
    failing = FakeSource(SourceName.ARXIV, error=SourceUnavailableError("arxiv", "503"))
    sources = [FakeSource(SourceName.PUBMED, [make_paper(SourceName.PUBMED, 1)]), failing]
    service = build_cached_service(sources, settings, cache)
    request = SearchRequest(query="prime editing")

    await service.search(request)
    assert cache.count() == 0

    failing._error = None  # the outage ends
    recovered = await service.search(request)
    assert recovered.degraded is False
    assert cache.count() == 1


# --- key sensitivity ----------------------------------------------------


async def test_different_queries_do_not_collide(
    settings: Settings, cache: SearchCache
) -> None:
    service = build_cached_service(healthy_sources(), settings, cache)
    await service.search(SearchRequest(query="prime editing"))
    other = await service.search(SearchRequest(query="base editing"))
    assert other.cache.hit is False


async def test_whitespace_and_case_do_not_defeat_the_cache(
    settings: Settings, cache: SearchCache
) -> None:
    """A re-typed query should hit; only real differences should miss."""
    service = build_cached_service(healthy_sources(), settings, cache)
    await service.search(SearchRequest(query="prime editing"))
    again = await service.search(SearchRequest(query="  Prime   Editing "))
    assert again.cache.hit is True


async def test_a_different_limit_is_a_different_search(
    settings: Settings, cache: SearchCache
) -> None:
    service = build_cached_service(healthy_sources(), settings, cache)
    await service.search(SearchRequest(query="prime editing", limit_per_source=10))
    wider = await service.search(SearchRequest(query="prime editing", limit_per_source=20))
    assert wider.cache.hit is False


async def test_changing_the_ranking_strategy_invalidates_the_cache(
    settings: Settings, cache: SearchCache
) -> None:
    """Config that changes the *answer* must change the key.

    The same reason the summary cache keys on prompt_version: serving a
    response produced under the old setting would silently undo the
    change.
    """
    request = SearchRequest(query="prime editing")
    await build_cached_service(healthy_sources(), settings, cache).search(request)

    changed = settings.model_copy(update={"ranking_strategy": "lexical"})
    after = await build_cached_service(healthy_sources(), changed, cache).search(request)
    assert after.cache.hit is False


# --- expiry and degradation of the cache itself -------------------------


async def test_an_expired_entry_is_a_miss_and_is_dropped(
    settings: Settings, tmp_path: Path
) -> None:
    expiring = SearchCache(tmp_path / "ttl.db", ttl_seconds=1)
    service = build_cached_service(healthy_sources(), settings, expiring)
    request = SearchRequest(query="prime editing")

    await service.search(request)
    assert expiring.count() == 1

    await asyncio.sleep(1.1)
    missed = await service.search(request)
    assert missed.cache.hit is False


async def test_no_cache_configured_still_searches(settings: Settings) -> None:
    """The cache is an optimisation; its absence must change nothing."""
    service = build_cached_service(healthy_sources(), settings, None)
    response = await service.search(SearchRequest(query="prime editing"))
    assert response.total > 0
    assert response.cache.hit is False
    assert response.cache.ttl_seconds is None


async def test_a_broken_cache_falls_through_to_a_live_search(
    settings: Settings, cache: SearchCache
) -> None:
    """A storage fault must cost latency, never results."""
    service = build_cached_service(healthy_sources(), settings, cache)
    cache.close()  # every call against it now raises

    response = await service.search(SearchRequest(query="prime editing"))
    assert response.total > 0
    assert response.cache.hit is False


# --- the key function in isolation --------------------------------------


def test_key_is_stable_and_order_independent_across_sources() -> None:
    a = cache_key("x", 20, [SourceName.PUBMED, SourceName.ARXIV], "cfg")
    b = cache_key("x", 20, [SourceName.ARXIV, SourceName.PUBMED], "cfg")
    assert a == b


def test_selecting_sources_differs_from_selecting_none() -> None:
    explicit = cache_key("x", 20, [SourceName.PUBMED], "cfg")
    everything = cache_key("x", 20, None, "cfg")
    assert explicit != everything
