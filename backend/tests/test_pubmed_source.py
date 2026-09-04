"""PubMed normalization tests.

MEDLINE XML is the most irregular of the three schemas, so these tests pin
down each irregularity the parser is expected to absorb.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.core.errors import SourceParseError
from app.models.paper import SourceName
from app.sources.pubmed import PubMedSource
from tests.conftest import load_fixture


@pytest.fixture
def papers(pubmed: PubMedSource) -> list:
    return pubmed.parse_articles(load_fixture("pubmed_efetch.xml"))


def test_parses_every_article_in_the_set(papers: list) -> None:
    assert len(papers) == 3
    assert [p.source for p in papers] == [SourceName.PUBMED] * 3


def test_maps_core_bibliographic_fields(papers: list) -> None:
    paper = papers[0]
    assert paper.title == "Prime editing efficiency in primary human cells."
    assert paper.source_id == "35123456"
    assert paper.doi == "10.1038/s41587-022-01234-5"
    assert paper.journal == "Nature Biotechnology"
    assert paper.volume == "40"
    assert paper.issue == "7"
    assert paper.pages == "1021-1030"
    assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/35123456/"


def test_id_is_doi_derived_when_a_doi_exists(papers: list) -> None:
    # A shared, DOI-derived id is what lets the same work collapse across
    # sources during deduplication.
    assert papers[0].id == "doi:10.1038/s41587-022-01234-5"
    assert papers[2].id.startswith("pubmed:")


def test_structured_abstract_sections_are_labelled_and_joined(papers: list) -> None:
    abstract = papers[0].abstract
    assert abstract is not None
    assert abstract.startswith("Background: Prime editing enables")
    assert "Methods: We benchmarked PE2 and PE3" in abstract  # inline <i> stripped
    assert "Results: Editing efficiency reached 42%" in abstract


def test_authors_keep_split_name_parts_and_affiliation(papers: list) -> None:
    first, second = papers[0].authors
    assert (first.given, first.family, first.name) == ("Wei", "Chen", "Wei Chen")
    assert first.affiliation == "Broad Institute, Cambridge, MA, USA."
    assert second.name == "Adaeze Okafor"
    assert second.affiliation is None


def test_collective_authors_are_preserved_as_a_single_name(papers: list) -> None:
    assert papers[1].authors[0].name == "The Genome Editing Consortium"


def test_initials_are_used_when_no_forename_is_present(papers: list) -> None:
    assert papers[2].authors[0].name == "JR Novak"


def test_prefers_the_electronic_article_date(papers: list) -> None:
    # ArticleDate is complete (2022-06-13); PubDate is only 2022-Jul.
    assert papers[0].published_date == date(2022, 6, 13)


def test_free_text_medline_date_is_parsed_to_year_and_month(papers: list) -> None:
    assert papers[1].published_date == date(2019, 1, 1)


def test_year_only_date_falls_back_to_january(papers: list) -> None:
    assert papers[2].published_date == date(2016, 1, 1)


def test_doi_is_recovered_from_the_article_id_list(papers: list) -> None:
    # Record 2 has no ELocationID; the DOI only exists in PubmedData.
    assert papers[1].doi == "10.1016/j.jmb.2019.01.002"


def test_missing_abstract_and_doi_are_none_not_empty_strings(papers: list) -> None:
    paper = papers[2]
    assert paper.abstract is None
    assert paper.doi is None
    assert paper.has_abstract is False


def test_inline_markup_is_stripped_from_titles(papers: list) -> None:
    assert papers[1].title == "Base editing outcomes in 13C-labelled cultures."


def test_author_keywords_win_over_mesh_terms(papers: list) -> None:
    assert papers[0].keywords == ["prime editing", "CRISPR"]


def test_mesh_terms_are_used_when_there_are_no_author_keywords(papers: list) -> None:
    assert papers[1].keywords == ["Gene Editing"]


def test_esearch_returns_the_pmid_list(pubmed: PubMedSource) -> None:
    assert pubmed.parse_esearch(load_fixture("pubmed_esearch.json")) == [
        "35123456",
        "31987654",
        "28001122",
    ]


def test_esearch_error_payload_raises_a_parse_error(pubmed: PubMedSource) -> None:
    with pytest.raises(SourceParseError, match="Empty term"):
        pubmed.parse_esearch(load_fixture("pubmed_esearch_error.json"))


def test_malformed_xml_raises_a_parse_error(pubmed: PubMedSource) -> None:
    with pytest.raises(SourceParseError):
        pubmed.parse_articles("<PubmedArticleSet><PubmedArticle>")


def test_empty_result_set_is_not_an_error(pubmed: PubMedSource) -> None:
    assert pubmed.parse_articles("<PubmedArticleSet/>") == []


def test_api_key_raises_the_rate_limit(settings) -> None:
    anonymous = PubMedSource(settings)
    with_key = PubMedSource(settings.model_copy(update={"pubmed_api_key": "abc123"}))
    assert with_key._rate_limit_interval() < anonymous._rate_limit_interval()
