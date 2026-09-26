"""Recover abstracts that the retrieving source never had.

Roughly one record in five comes back with no abstract — overwhelmingly
from Crossref, where depositing one is optional and many publishers do
not. Those papers are the weakest thing in the pipeline: they cannot be
summarized, cannot be grounded, and rank on a title and a few keywords
while their neighbours rank on a paragraph.

The abstract usually exists; it is just not in the record we received.
OpenAlex holds abstracts for a large share of DOIs Crossref itself lacks,
so this asks for them.

**Measured on the golden set before building it:** 77 of 411 pooled
records had no abstract, and OpenAlex returned one for 32% of a sampled
25. That is the honest ceiling — the other two thirds are editorials,
book chapters and conference front-matter that never had an abstract
anywhere. This closes about a third of the gap, not the gap.

Two things keep it cheap enough to sit in the request path:

* **Batching.** OpenAlex filters on up to 50 DOIs per call, so a search
  with 77 gaps costs two requests rather than 77. Measured at ~1.4s for a
  batch.
* **It cannot fail the search.** Any error, timeout or malformed payload
  leaves the papers exactly as they arrived. An enrichment that can take
  down a result set is not worth having.

Abstracts arrive as an *inverted index* — ``{word: [positions]}`` — which
is OpenAlex's way of shipping text it cannot redistribute verbatim. It
reconstructs exactly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import httpx

from app.config import Settings
from app.core.logging import get_logger
from app.core.text import collapse_whitespace, normalize_doi
from app.models.paper import Paper
from app.models.search import BackfillReport

logger = get_logger(__name__)

#: OpenAlex accepts up to 50 values in an OR filter.
_BATCH = 50

#: A reconstructed "abstract" shorter than this is a stub — a copyright
#: line or a single sentence of front-matter — and is worse than nothing,
#: because it would make the paper look summarizable when it is not.
_MIN_LENGTH = 120


class AbstractBackfill:
    """Fills in missing abstracts from OpenAlex, or leaves papers untouched."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._base_url = settings.openalex_base_url
        # OpenAlex asks callers to identify themselves and serves the
        # polite pool faster in return. Falling back to the Crossref
        # address avoids a second setting for the same fact about the
        # operator.
        self._mailto = settings.openalex_mailto or settings.crossref_mailto
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.abstract_backfill_timeout_seconds),
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fill(self, papers: Sequence[Paper]) -> tuple[list[Paper], BackfillReport]:
        """Return the papers with any recoverable abstracts filled in."""
        started = time.perf_counter()
        gaps = [p for p in papers if p.doi and not p.abstract]
        if not gaps:
            return list(papers), BackfillReport(
                applied=False, reason="every record already has an abstract"
            )

        batches = [gaps[i : i + _BATCH] for i in range(0, len(gaps), _BATCH)]
        results = await asyncio.gather(
            *(self._fetch(batch) for batch in batches), return_exceptions=True
        )

        found: dict[str, str] = {}
        failures = 0
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                logger.warning("abstract backfill batch failed: %s", result)
                continue
            found.update(result)

        filled = [
            paper.model_copy(update={"abstract": found[paper.doi]})
            if paper.doi and not paper.abstract and paper.doi in found
            else paper
            for paper in papers
        ]

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        report = BackfillReport(
            applied=True,
            source="openalex",
            missing=len(gaps),
            recovered=len(found),
            elapsed_ms=elapsed_ms,
            reason=(
                f"{failures} of {len(batches)} batches failed" if failures else None
            ),
        )
        logger.info(
            "abstract backfill: recovered %d of %d in %dms", len(found), len(gaps), elapsed_ms
        )
        return filled, report

    async def _fetch(self, batch: Sequence[Paper]) -> dict[str, str]:
        """One batched lookup. Returns ``{doi: abstract}`` for what it found."""
        dois = [p.doi for p in batch if p.doi]
        params = {
            "filter": "doi:" + "|".join(dois),
            "per-page": str(len(dois)),
            # Ask only for the two fields used. The full work record is
            # large, and a search may be fetching fifty of them.
            "select": "doi,abstract_inverted_index",
        }
        if self._mailto:
            params["mailto"] = self._mailto

        response = await self._client.get(f"{self._base_url}/works", params=params)
        response.raise_for_status()
        payload = response.json()

        out: dict[str, str] = {}
        for work in payload.get("results", []):
            doi = normalize_doi(work.get("doi"))
            text = reconstruct_abstract(work.get("abstract_inverted_index"))
            if doi and text and len(text) >= _MIN_LENGTH:
                out[doi] = text
        return out


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str | None:
    """Rebuild running text from OpenAlex's inverted index.

    ``{"The": [0], "cat": [1, 4]}`` becomes ``"The cat ... cat"``. Positions
    are authoritative; a word can appear at several of them, and gaps are
    possible when a token was dropped upstream, so this sorts by position
    rather than assuming a dense range.
    """
    if not index:
        return None
    positions: dict[int, str] = {}
    for word, spots in index.items():
        if not isinstance(spots, list):
            continue
        for spot in spots:
            if isinstance(spot, int):
                positions[spot] = word
    if not positions:
        return None
    return collapse_whitespace(" ".join(positions[i] for i in sorted(positions)))
