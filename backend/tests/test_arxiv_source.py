"""arXiv normalization tests."""

from __future__ import annotations

from datetime import date

import pytest

from app.core.errors import SourceParseError
from app.models.paper import SourceName
from app.sources.arxiv import ArxivSource
from tests.conftest import load_fixture


@pytest.fixture
def papers(arxiv: ArxivSource) -> list:
    return arxiv.parse_feed(load_fixture("arxiv_feed.xml"))


def test_parses_every_entry(papers: list) -> None:
    assert len(papers) == 2
    assert all(p.source is SourceName.ARXIV for p in papers)


def test_maps_core_fields(papers: list) -> None:
    paper = papers[0]
    assert paper.title == "Attention Is All You Need"
    assert paper.abstract is not None
    assert paper.abstract.startswith("The dominant sequence transduction models")
    assert paper.published_date == date(2017, 6, 12)
    assert paper.publisher == "arXiv"
    assert paper.publication_type == "preprint"


def test_version_suffix_is_stripped_from_the_identifier(papers: list) -> None:
    # The versioned id changes on every revision; the bare id is stable.
    assert papers[0].source_id == "1706.03762"


def test_records_without_a_doi_fall_back_to_a_source_scoped_id(papers: list) -> None:
    assert papers[0].doi is None
    assert papers[0].id.startswith("arxiv:")


def test_published_preprints_expose_their_journal_doi(papers: list) -> None:
    paper = papers[1]
    assert paper.doi == "10.1038/s41587-022-01234-5"
    assert paper.id == "doi:10.1038/s41587-022-01234-5"
    assert paper.journal == "Nat Biotechnol 40, 1021-1030 (2022)"


def test_multiline_titles_are_collapsed_to_one_line(papers: list) -> None:
    assert papers[1].title == "Prime editing efficiency in primary human cells"


def test_authors_are_split_into_given_and_family_names(papers: list) -> None:
    first = papers[0].authors[0]
    assert (first.name, first.given, first.family) == ("Ashish Vaswani", "Ashish", "Vaswani")
    assert first.affiliation == "Google Brain"


def test_categories_become_keywords(papers: list) -> None:
    assert papers[0].keywords == ["cs.CL", "cs.LG"]


def test_pdf_link_is_extracted(papers: list) -> None:
    assert papers[0].pdf_url == "http://arxiv.org/pdf/1706.03762v7"


def test_error_feed_raises_instead_of_returning_an_empty_list(arxiv: ArxivSource) -> None:
    # arXiv answers malformed queries with HTTP 200 and an error entry, so
    # a status-code-only check would silently report "no results".
    with pytest.raises(SourceParseError, match="incorrect id format"):
        arxiv.parse_feed(load_fixture("arxiv_error.xml"))


def test_malformed_xml_raises_a_parse_error(arxiv: ArxivSource) -> None:
    with pytest.raises(SourceParseError):
        arxiv.parse_feed("not xml at all")


@pytest.mark.parametrize(
    ("terms", "expected"),
    [
        ("crispr", 'all:"crispr"'),
        ("prime editing", 'all:"prime" AND all:"editing"'),
        ('quotes "and" backslashes\\', 'all:"quotes" AND all:"and" AND all:"backslashes"'),
        ("   ", "all:*"),
    ],
)
def test_search_query_ands_terms_rather_than_phrase_matching(
    arxiv: ArxivSource, terms: str, expected: str
) -> None:
    assert arxiv._build_search_query(terms) == expected
