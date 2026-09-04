"""Crossref normalization tests."""

from __future__ import annotations

from datetime import date

import pytest

from app.core.errors import SourceParseError
from app.models.paper import SourceName
from app.sources.crossref import CrossrefSource
from tests.conftest import load_fixture


@pytest.fixture
def papers(crossref: CrossrefSource) -> list:
    return crossref.parse_work_list(load_fixture("crossref_works.json"))


def test_untitled_records_are_dropped(papers: list) -> None:
    # The fixture holds four items; the component record has no title and
    # cannot be ranked, cited, or displayed.
    assert len(papers) == 3
    assert all(p.source is SourceName.CROSSREF for p in papers)


def test_maps_core_fields(papers: list) -> None:
    paper = papers[0]
    assert paper.title == "Prime editing efficiency in primary human cells"
    assert paper.journal == "Nature Biotechnology"
    assert paper.publisher == "Springer Science and Business Media LLC"
    assert (paper.volume, paper.issue, paper.pages) == ("40", "7", "1021-1030")
    assert paper.citation_count == 214
    assert paper.pdf_url == "https://www.nature.com/articles/s41587-022-01234-5.pdf"


def test_dois_are_lowercased_for_cross_source_matching(papers: list) -> None:
    # Crossref returns this DOI uppercased; PubMed returns it lowercased.
    assert papers[0].doi == "10.1038/s41587-022-01234-5"
    assert papers[0].id == "doi:10.1038/s41587-022-01234-5"


def test_jats_markup_is_stripped_from_abstracts(papers: list) -> None:
    assert papers[0].abstract == (
        "Prime editing enables precise genome edits without double-strand breaks."
    )


def test_plain_text_abstracts_pass_through(papers: list) -> None:
    assert papers[2].abstract == "Plain text abstract with no JATS wrapper at all."


def test_ragged_date_parts_fall_back_across_date_fields(papers: list) -> None:
    # issued is [[null]], so published-print supplies the date.
    assert papers[1].published_date == date(2020, 5, 14)


def test_partial_dates_are_padded_rather_than_discarded(papers: list) -> None:
    assert papers[0].published_date == date(2022, 7, 1)  # year + month
    assert papers[2].published_date == date(2021, 1, 1)  # year only


def test_missing_abstract_is_none(papers: list) -> None:
    assert papers[1].abstract is None
    assert papers[1].has_abstract is False


def test_empty_container_title_array_becomes_none(papers: list) -> None:
    assert papers[1].journal is None


def test_organisation_authors_are_kept_as_a_single_name(papers: list) -> None:
    author = papers[1].authors[0]
    assert author.name == "The COVID-19 Genomics Consortium"


def test_author_affiliation_is_captured(papers: list) -> None:
    assert papers[0].authors[0].affiliation == "Broad Institute"
    assert papers[0].authors[1].affiliation is None


def test_subjects_become_keywords(papers: list) -> None:
    assert papers[0].keywords == ["Biomedical Engineering", "Molecular Medicine"]


def test_invalid_json_raises_a_parse_error(crossref: CrossrefSource) -> None:
    with pytest.raises(SourceParseError, match="invalid JSON"):
        crossref.parse_work_list("<html>502 Bad Gateway</html>")


def test_unexpected_shape_raises_a_parse_error(crossref: CrossrefSource) -> None:
    with pytest.raises(SourceParseError, match="message.items"):
        crossref.parse_work_list('{"status": "ok", "message": {}}')
