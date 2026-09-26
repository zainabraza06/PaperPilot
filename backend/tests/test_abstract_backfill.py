"""Recovering abstracts the retrieving source never had.

Crossref makes abstracts optional and many publishers skip them, so
roughly one pooled record in five arrives with nothing to rank, summarize
or ground against. OpenAlex has the abstract for about a third of those.

These tests pin the behaviour with a mocked OpenAlex, because the point
of the backfill is not that the network works — it is that a paper gains
the right abstract, that nothing else is touched, and that a failure
costs nothing.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.models.paper import Paper, SourceName
from app.services.enrichment.abstracts import AbstractBackfill, reconstruct_abstract

pytestmark = pytest.mark.asyncio

OPENALEX = "https://api.openalex.org/works"

#: Long enough to clear the stub threshold the service applies.
LONG = (
    "We report a method for recovering abstracts that the retrieving source "
    "never held, and evaluate it against a frozen pool of real records drawn "
    "from three separate providers over several months."
)


def inverted(text: str) -> dict[str, list[int]]:
    """Encode text the way OpenAlex ships it."""
    index: dict[str, list[int]] = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


def paper(n: int, *, doi: str | None, abstract: str | None = None) -> Paper:
    """A Crossref-shaped record.

    The DOIs are deliberately well-formed. ``Paper`` normalises its DOI on
    validation and a registrant code outside 4-9 digits is dropped to
    ``None`` — so a toy value like "10.1/a" silently becomes a paper with
    no DOI, and every assertion here quietly passes for the wrong reason.
    """
    return Paper(
        id=f"p{n}",
        doi=doi,
        source=SourceName.CROSSREF,
        source_id=str(n),
        title=f"paper {n}",
        abstract=abstract,
        url=f"https://example.org/{n}",
    )


def work(doi: str, text: str | None) -> dict[str, object]:
    return {
        "doi": f"https://doi.org/{doi}",
        "abstract_inverted_index": inverted(text) if text else None,
    }


@pytest.fixture
def backfill(settings: Settings) -> AbstractBackfill:
    return AbstractBackfill(settings)


# --- the happy path -----------------------------------------------------


@respx.mock
async def test_a_missing_abstract_is_filled_in(backfill: AbstractBackfill) -> None:
    respx.get(OPENALEX).mock(
        return_value=httpx.Response(200, json={"results": [work("10.5555/a", LONG)]})
    )
    filled, report = await backfill.fill([paper(1, doi="10.5555/a")])

    assert filled[0].abstract == LONG
    assert report.applied is True
    assert (report.missing, report.recovered) == (1, 1)
    assert report.source == "openalex"


@respx.mock
async def test_an_existing_abstract_is_never_overwritten(
    backfill: AbstractBackfill,
) -> None:
    """The retrieved record is authoritative; this only fills gaps."""
    route = respx.get(OPENALEX).mock(
        return_value=httpx.Response(200, json={"results": [work("10.5555/a", LONG)]})
    )
    original = "The abstract the source actually returned."
    filled, report = await backfill.fill([paper(1, doi="10.5555/a", abstract=original)])

    assert filled[0].abstract == original
    assert report.applied is False
    assert not route.called, "no request should be made when there is nothing to fill"


@respx.mock
async def test_only_the_gaps_are_requested(backfill: AbstractBackfill) -> None:
    route = respx.get(OPENALEX).mock(
        return_value=httpx.Response(200, json={"results": [work("10.5555/b", LONG)]})
    )
    await backfill.fill(
        [
            paper(1, doi="10.5555/a", abstract="already here and quite long indeed"),
            paper(2, doi="10.5555/b"),
            paper(3, doi=None),  # nothing to look up with
        ]
    )
    sent = str(route.calls[0].request.url)
    assert "10.5555%2Fb" in sent or "10.5555/b" in sent
    assert "10.5555/a" not in sent and "10.5555%2Fa" not in sent


@respx.mock
async def test_a_paper_with_no_doi_is_left_alone(backfill: AbstractBackfill) -> None:
    route = respx.get(OPENALEX).mock(return_value=httpx.Response(200, json={"results": []}))
    filled, report = await backfill.fill([paper(1, doi=None)])
    assert filled[0].abstract is None
    assert report.applied is False
    assert not route.called


# --- batching -----------------------------------------------------------


@respx.mock
async def test_large_gaps_are_batched_rather_than_requested_one_by_one(
    backfill: AbstractBackfill,
) -> None:
    """120 gaps must cost three requests, not 120.

    This is what makes the feature affordable inside a request: a search
    with eighty missing abstracts should add a second, not a minute.
    """
    route = respx.get(OPENALEX).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await backfill.fill([paper(n, doi=f"10.5555/{n}") for n in range(120)])
    assert route.call_count == 3


# --- what must not be accepted -----------------------------------------


@respx.mock
async def test_a_stub_abstract_is_rejected(backfill: AbstractBackfill) -> None:
    """A copyright line is worse than nothing.

    An empty abstract is honestly reported as "no abstract deposited". A
    twenty-character stub makes the paper look summarizable and grounded
    when neither is true.
    """
    respx.get(OPENALEX).mock(
        return_value=httpx.Response(200, json={"results": [work("10.5555/a", "© 2024 Elsevier.")]})
    )
    filled, report = await backfill.fill([paper(1, doi="10.5555/a")])
    assert filled[0].abstract is None
    assert report.recovered == 0


@respx.mock
async def test_a_work_with_no_abstract_is_simply_not_filled(
    backfill: AbstractBackfill,
) -> None:
    respx.get(OPENALEX).mock(
        return_value=httpx.Response(200, json={"results": [work("10.5555/a", None)]})
    )
    filled, report = await backfill.fill([paper(1, doi="10.5555/a")])
    assert filled[0].abstract is None
    assert (report.missing, report.recovered) == (1, 0)


# --- failure must cost nothing -----------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(500),
        httpx.Response(429),
        httpx.Response(200, text="not json at all"),
    ],
)
@respx.mock
async def test_an_upstream_failure_leaves_the_papers_untouched(
    backfill: AbstractBackfill, failure: httpx.Response
) -> None:
    respx.get(OPENALEX).mock(return_value=failure)
    papers = [paper(1, doi="10.5555/a")]
    filled, report = await backfill.fill(papers)

    assert filled[0].abstract is None
    assert report.recovered == 0
    assert report.reason is not None, "a silent failure would be worse than a slow one"


@respx.mock
async def test_a_timeout_leaves_the_papers_untouched(
    backfill: AbstractBackfill,
) -> None:
    respx.get(OPENALEX).mock(side_effect=httpx.ReadTimeout("too slow"))
    filled, report = await backfill.fill([paper(1, doi="10.5555/a")])
    assert filled[0].abstract is None
    assert report.recovered == 0


@respx.mock
async def test_one_failed_batch_does_not_discard_the_others(
    backfill: AbstractBackfill,
) -> None:
    """Partial recovery beats none."""
    respx.get(OPENALEX).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json={"results": [work("10.5555/60", LONG)]}),
            httpx.Response(200, json={"results": []}),
        ]
    )
    papers = [paper(n, doi=f"10.5555/{n}") for n in range(120)]
    filled, report = await backfill.fill(papers)

    assert report.recovered == 1
    assert next(p for p in filled if p.doi == "10.5555/60").abstract == LONG
    assert report.reason is not None and "failed" in report.reason


# --- the inverted index -------------------------------------------------


def test_the_inverted_index_round_trips() -> None:
    text = "the cat sat on the mat"
    assert reconstruct_abstract(inverted(text)) == text


def test_a_repeated_word_lands_at_every_position() -> None:
    assert reconstruct_abstract({"a": [0, 2], "b": [1]}) == "a b a"


def test_a_gap_in_the_positions_does_not_break_reconstruction() -> None:
    """Positions are authoritative and need not be dense."""
    assert reconstruct_abstract({"start": [0], "end": [7]}) == "start end"


@pytest.mark.parametrize("bad", [None, {}, {"a": "not a list"}, {"a": ["x"]}])
def test_a_malformed_index_yields_nothing_rather_than_raising(bad: object) -> None:
    assert reconstruct_abstract(bad) is None  # type: ignore[arg-type]
