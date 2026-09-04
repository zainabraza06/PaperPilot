"""Query classification tests.

The single search box promises four input modes. Misreading one of them is
user-visible: a pasted abstract sent verbatim to PubMed returns nothing,
and a DOI treated as a topic returns the wrong paper.
"""

from __future__ import annotations

import pytest

from app.models.search import IdentifierKind, QueryIntent
from app.services.query_parser import extract_keywords, parse_query

ABSTRACT = (
    "Prime editing enables precise genome edits without requiring double-strand breaks "
    "or donor DNA templates. Here we benchmark prime editing efficiency across six "
    "primary human cell types and show that editing efficiency reaches 42 percent in "
    "T cells while maintaining a low indel rate across all tested loci."
)


@pytest.mark.parametrize(
    "raw",
    [
        "10.1038/s41587-022-01234-5",
        "https://doi.org/10.1038/s41587-022-01234-5",
        "doi:10.1038/S41587-022-01234-5",
        "  10.1038/s41587-022-01234-5  ",
    ],
)
def test_dois_are_detected_and_normalized(raw: str) -> None:
    parsed = parse_query(raw)
    assert parsed.intent is QueryIntent.IDENTIFIER
    assert parsed.identifier_kind is IdentifierKind.DOI
    assert parsed.identifier == "10.1038/s41587-022-01234-5"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2101.00001", "2101.00001"),
        ("2101.00001v3", "2101.00001"),
        ("arXiv:1706.03762", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762v7", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
        ("math.GT/0309136", "math.GT/0309136"),
    ],
)
def test_arxiv_identifiers_are_detected(raw: str, expected: str) -> None:
    parsed = parse_query(raw)
    assert parsed.identifier_kind is IdentifierKind.ARXIV
    assert parsed.identifier == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("35123456", "35123456"),
        ("PMID: 35123456", "35123456"),
        ("https://pubmed.ncbi.nlm.nih.gov/35123456/", "35123456"),
    ],
)
def test_pmids_are_detected(raw: str, expected: str) -> None:
    parsed = parse_query(raw)
    assert parsed.identifier_kind is IdentifierKind.PMID
    assert parsed.identifier == expected


def test_single_words_are_keyword_queries() -> None:
    assert parse_query("CRISPR").intent is QueryIntent.KEYWORD


def test_short_phrases_are_topic_queries() -> None:
    assert parse_query("prime editing in T cells").intent is QueryIntent.TOPIC


def test_long_text_is_treated_as_a_pasted_abstract() -> None:
    parsed = parse_query(ABSTRACT)
    assert parsed.intent is QueryIntent.ABSTRACT_SNIPPET


def test_pasted_abstracts_are_distilled_into_an_api_friendly_query() -> None:
    # No upstream API accepts a paragraph, so the fan-out gets keywords...
    parsed = parse_query(ABSTRACT)
    assert len(parsed.search_terms.split()) <= 10
    assert "editing" in parsed.search_terms
    # ...while the full text survives for semantic ranking in Stage 2.
    assert parsed.raw == ABSTRACT


def test_a_doi_inside_prose_does_not_become_a_lookup() -> None:
    parsed = parse_query("papers citing 10.1038/s41587-022-01234-5 about prime editing")
    assert parsed.intent is not QueryIntent.IDENTIFIER


def test_keyword_extraction_drops_stopwords_and_abstract_filler() -> None:
    keywords = extract_keywords("The results show that we used a novel CRISPR screen")
    assert "the" not in keywords
    assert "results" not in keywords  # abstract filler
    assert "crispr" in keywords


def test_keyword_extraction_ranks_by_frequency_then_first_appearance() -> None:
    keywords = extract_keywords("genome genome editing genome editing repair")
    assert keywords[:3] == ["genome", "editing", "repair"]


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        parse_query("   ")
