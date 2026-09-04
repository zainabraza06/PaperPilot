"""Deduplication tests.

Showing the same paper three times is the most visible failure mode of a
multi-source search, and a *false* merge silently hides a paper — so both
directions are pinned down here.
"""

from __future__ import annotations

from datetime import date

from app.models.paper import Author, Paper, SourceName
from app.services.dedupe import deduplicate


def make_paper(
    *,
    source: SourceName,
    title: str,
    doi: str | None = None,
    abstract: str | None = None,
    source_id: str = "x",
    **kwargs: object,
) -> Paper:
    return Paper(
        id=f"{source.value}:{source_id}",
        doi=doi,
        source=source,
        source_id=source_id,
        title=title,
        abstract=abstract,
        url=f"https://example.org/{source_id}",
        **kwargs,
    )


def test_records_sharing_a_doi_are_merged() -> None:
    papers = [
        make_paper(source=SourceName.PUBMED, title="A study", doi="10.1234/abc", source_id="1"),
        make_paper(source=SourceName.CROSSREF, title="A study", doi="10.1234/ABC", source_id="2"),
    ]
    result = deduplicate(papers)
    assert result.duplicates_merged == 1
    assert len(result.papers) == 1


def test_merged_record_reports_every_contributing_source() -> None:
    papers = [
        make_paper(
            source=SourceName.PUBMED, title="A study", doi="10.1234/abc", abstract="Full text."
        ),
        make_paper(source=SourceName.CROSSREF, title="A study", doi="10.1234/abc"),
        make_paper(source=SourceName.ARXIV, title="A study", doi="10.1234/abc"),
    ]
    merged = deduplicate(papers).papers[0]
    assert merged.source is SourceName.PUBMED
    assert set(merged.also_found_in) == {SourceName.CROSSREF, SourceName.ARXIV}
    assert set(merged.all_sources) == set(SourceName)


def test_the_most_complete_record_wins_the_identity() -> None:
    sparse = make_paper(source=SourceName.CROSSREF, title="A study", doi="10.1234/abc")
    rich = make_paper(
        source=SourceName.ARXIV,
        title="A study",
        doi="10.1234/abc",
        abstract="An abstract, which dominates the completeness score.",
        published_date=date(2021, 1, 1),
        authors=[Author(name="Ada Lovelace")],
    )
    merged = deduplicate([sparse, rich]).papers[0]
    assert merged.source is SourceName.ARXIV
    assert merged.abstract is not None


def test_fields_are_filled_from_siblings_rather_than_lost() -> None:
    # The real payoff: Crossref has the citation count and journal, arXiv
    # has the abstract. The user should see all of it on one card.
    crossref = make_paper(
        source=SourceName.CROSSREF,
        title="A study",
        doi="10.1234/abc",
        journal="Nature",
        citation_count=99,
        published_date=date(2022, 3, 1),
        authors=[Author(name="Ada Lovelace")],
    )
    arxiv = make_paper(
        source=SourceName.ARXIV,
        title="A study",
        doi="10.1234/abc",
        abstract="The abstract that Crossref never received.",
        pdf_url="https://arxiv.org/pdf/1234",
    )
    merged = deduplicate([crossref, arxiv]).papers[0]
    assert merged.abstract == "The abstract that Crossref never received."
    assert merged.journal == "Nature"
    assert merged.citation_count == 99
    assert merged.pdf_url == "https://arxiv.org/pdf/1234"
    assert merged.authors[0].name == "Ada Lovelace"


def test_doiless_records_merge_on_a_normalized_title() -> None:
    papers = [
        make_paper(source=SourceName.ARXIV, title="Attention Is All You Need", source_id="1"),
        make_paper(source=SourceName.PUBMED, title="Attention is all you need.", source_id="2"),
    ]
    assert len(deduplicate(papers).papers) == 1


def test_near_identical_titles_merge_through_the_fuzzy_pass() -> None:
    papers = [
        make_paper(
            source=SourceName.ARXIV,
            title="Deep Residual Learning for Image Recognition",
            source_id="1",
        ),
        make_paper(
            source=SourceName.CROSSREF,
            title="Deep Residual Learning for Image Recognitions",
            source_id="2",
        ),
    ]
    assert len(deduplicate(papers).papers) == 1


def test_a_doi_recovered_from_a_sibling_updates_the_canonical_id() -> None:
    papers = [
        make_paper(source=SourceName.ARXIV, title="Same work", source_id="1"),
        make_paper(source=SourceName.CROSSREF, title="Same work", doi="10.5678/xyz", source_id="2"),
    ]
    merged = deduplicate(papers).papers[0]
    assert merged.doi == "10.5678/xyz"
    assert merged.id == "doi:10.5678/xyz"


def test_different_papers_are_not_merged() -> None:
    papers = [
        make_paper(source=SourceName.ARXIV, title="Attention Is All You Need", source_id="1"),
        make_paper(source=SourceName.ARXIV, title="Language Models Are Few-Shot Learners", source_id="2"),
        make_paper(source=SourceName.PUBMED, title="CRISPR screens in primary T cells", source_id="3"),
    ]
    result = deduplicate(papers)
    assert len(result.papers) == 3
    assert result.duplicates_merged == 0


def test_papers_with_the_same_prefix_but_different_topics_stay_separate() -> None:
    # Guards the blocking key: both titles share their first 12 characters.
    papers = [
        make_paper(source=SourceName.ARXIV, title="A survey of graph neural networks", source_id="1"),
        make_paper(source=SourceName.ARXIV, title="A survey of reinforcement learning", source_id="2"),
    ]
    assert len(deduplicate(papers).papers) == 2


def test_conflicting_dois_prevent_a_title_merge() -> None:
    # Errata and corrections often share a title with the original paper.
    papers = [
        make_paper(source=SourceName.CROSSREF, title="A study", doi="10.1234/abc", source_id="1"),
        make_paper(source=SourceName.CROSSREF, title="A study", doi="10.1234/def", source_id="2"),
    ]
    result = deduplicate(papers)
    assert len(result.papers) == 2


def test_input_order_is_preserved() -> None:
    papers = [
        make_paper(source=SourceName.ARXIV, title=f"Paper {n}", source_id=str(n)) for n in range(5)
    ]
    assert [p.title for p in deduplicate(papers).papers] == [f"Paper {n}" for n in range(5)]


def test_empty_input_is_handled() -> None:
    result = deduplicate([])
    assert result.papers == []
    assert result.duplicates_merged == 0
