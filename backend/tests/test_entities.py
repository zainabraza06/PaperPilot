"""Entity extraction tests.

The pattern pass and the trust guards are pure functions and are tested
directly. The spaCy pass is tested against a stub pipeline rather than a
real model: what matters is that label normalization, the trust rules and
the span bookkeeping are correct, and pinning those to a downloaded
model's exact predictions would make the suite slow and brittle.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from app.models.entities import EntityLabel
from app.models.paper import Paper, SourceName
from app.services.enrichment.entities import (
    PatternEntityExtractor,
    SpacyEntityExtractor,
    _trust_general,
    build_entity_extractor,
)


def make_paper(title: str, abstract: str | None = None, paper_id: str = "p1") -> Paper:
    return Paper(
        id=paper_id,
        source=SourceName.ARXIV,
        source_id="1",
        title=title,
        abstract=abstract,
        url="https://example.org/1",
    )


# --- pattern pass --------------------------------------------------------


def extract_texts(title: str, abstract: str | None = None) -> dict[str, EntityLabel]:
    entities = PatternEntityExtractor().extract([make_paper(title, abstract)])[0]
    return {entity.text: entity.label for entity in entities}


@pytest.mark.parametrize(
    "term",
    ["CRISPR", "DNA", "pegRNA", "sgRNA", "BRCA1", "PE3", "SARS-CoV-2", "CRISPR-Cas9", "HEK293T"],
)
def test_symbol_shaped_terms_are_detected(term: str) -> None:
    # These are exactly what a general NER model misses and a researcher scans for.
    assert term in extract_texts(f"A study of {term} in cells")


@pytest.mark.parametrize("assay", ["RNA-seq", "scRNA-seq", "ChIP-seq", "ATAC-seq"])
def test_assay_names_are_detected(assay: str) -> None:
    assert assay in extract_texts(f"Profiling with {assay}")


def test_ordinary_words_are_not_entities() -> None:
    found = extract_texts("The study of cells in the human body was conducted")
    assert found == {}


def test_structured_abstract_headings_are_not_entities() -> None:
    # PubMed abstracts are re-assembled with labels like "METHODS:", which
    # match the acronym shape but are formatting, not content.
    found = extract_texts("A study", "BACKGROUND: we did things. RESULTS: it worked. CONCLUSIONS: good.")
    assert found == {}


def test_repeated_mentions_collapse_into_one_entity_with_every_span() -> None:
    entities = PatternEntityExtractor().extract(
        [make_paper("CRISPR editing", "CRISPR is useful. We used CRISPR again.")]
    )[0]
    crispr = next(e for e in entities if e.text == "CRISPR")
    assert crispr.count == 3
    assert crispr.in_title is True
    assert {span.field for span in crispr.spans} == {"title", "abstract"}


def test_spans_point_at_the_actual_text() -> None:
    # The frontend highlights inline using these offsets, so an off-by-one
    # here is a visibly broken UI.
    abstract = "We applied pegRNA to the problem."
    entities = PatternEntityExtractor().extract([make_paper("A study", abstract)])[0]
    entity = next(e for e in entities if e.text == "pegRNA")
    span = entity.spans[0]
    assert abstract[span.start : span.end] == "pegRNA"


def test_papers_without_abstracts_still_yield_title_entities() -> None:
    assert "CRISPR" in extract_texts("CRISPR screening", None)


def test_extracting_no_papers_returns_no_lists() -> None:
    assert PatternEntityExtractor().extract([]) == []


def test_output_is_one_list_per_input_paper_in_order() -> None:
    papers = [make_paper("CRISPR", paper_id="a"), make_paper("no entities here", paper_id="b")]
    results = PatternEntityExtractor().extract(papers)
    assert len(results) == 2
    assert results[0][0].text == "CRISPR"
    assert results[1] == []


# --- trust guards for a general-purpose model -----------------------------


@pytest.mark.parametrize(
    ("label", "text", "field", "trusted"),
    [
        # Institutions in an abstract are what the news model is good at.
        ("ORG", "Broad Institute", "abstract", True),
        ("ORG", "Google Brain", "abstract", True),
        ("PERSON", "Ada Lovelace", "abstract", True),
        # Techniques the model mislabels as organizations.
        ("ORG", "CRISPR-Cas9", "abstract", False),
        ("ORG", "HDR", "abstract", False),
        ("ORG", "pegRNA", "abstract", False),
        # A single-word ORG in an abstract is a tool, not an institution.
        ("ORG", "Transformer", "abstract", False),
        ("ORG", "Stanford University", "abstract", True),
        # Title-case fools the model into seeing proper nouns everywhere.
        ("ORG", "Broad Institute", "title", False),
        # Labels the model has no business emitting on this text.
        ("GPE", "HEK293", "abstract", False),
        ("LAW", "pegRNA improves CRISPR-Cas9", "abstract", False),
    ],
)
def test_general_model_predictions_are_trusted_selectively(
    label: str, text: str, field: str, trusted: bool
) -> None:
    assert _trust_general(label, text, field) is trusted  # type: ignore[arg-type]


# --- spaCy pass, against a stub pipeline ----------------------------------


class StubEnt(NamedTuple):
    text: str
    label_: str
    start_char: int
    end_char: int


class StubDoc:
    def __init__(self, ents: list[StubEnt]) -> None:
        self.ents = ents


class StubPipeline:
    """Stands in for a loaded spaCy model, returning scripted entities."""

    def __init__(self, by_text: dict[str, list[StubEnt]]) -> None:
        self._by_text = by_text

    def pipe(self, texts: list[str], batch_size: int = 32) -> list[StubDoc]:
        return [StubDoc(self._by_text.get(text, [])) for text in texts]


def build_stub_extractor(
    by_text: dict[str, list[StubEnt]], *, scientific: bool
) -> SpacyEntityExtractor:
    extractor = SpacyEntityExtractor("stub-model")
    extractor._nlp = StubPipeline(by_text)
    extractor._scientific = scientific
    extractor.model_id = "stub-model"
    return extractor


def test_scispacy_labels_are_normalized_onto_our_vocabulary() -> None:
    abstract = "TP53 mutations cause carcinoma treated with cisplatin."
    extractor = build_stub_extractor(
        {
            "A study": [],
            abstract: [
                StubEnt("TP53", "GENE_OR_GENE_PRODUCT", 0, 4),
                StubEnt("carcinoma", "DISEASE", 21, 30),
                StubEnt("cisplatin", "CHEMICAL", 44, 53),
            ],
        },
        scientific=True,
    )
    found = {e.text: e.label for e in extractor.extract([make_paper("A study", abstract)])[0]}
    assert found["TP53"] is EntityLabel.GENE_OR_PROTEIN
    assert found["carcinoma"] is EntityLabel.DISEASE
    assert found["cisplatin"] is EntityLabel.CHEMICAL


def test_numeric_labels_are_discarded_whatever_the_model() -> None:
    abstract = "In 2022 we saw 42% improvement."
    extractor = build_stub_extractor(
        {"A study": [], abstract: [StubEnt("2022", "DATE", 3, 7), StubEnt("42%", "PERCENT", 15, 18)]},
        scientific=True,
    )
    assert extractor.extract([make_paper("A study", abstract)])[0] == []


def test_leading_articles_are_stripped_from_organisations() -> None:
    abstract = "Work done at the Broad Institute today."
    extractor = build_stub_extractor(
        {"A study": [], abstract: [StubEnt("the Broad Institute", "ORG", 13, 32)]},
        scientific=False,
    )
    entity = extractor.extract([make_paper("A study", abstract)])[0][0]
    assert entity.text == "Broad Institute"
    # The span must still line up with the text after the article is dropped.
    span = entity.spans[0]
    assert abstract[span.start : span.end] == "Broad Institute"


def test_the_pattern_pass_fills_spans_the_model_did_not_claim() -> None:
    # The model claims only the institution; pegRNA is left to the patterns.
    abstract = "Broad Institute used pegRNA here."
    extractor = build_stub_extractor(
        {"A study": [], abstract: [StubEnt("Broad Institute", "ORG", 0, 15)]},
        scientific=False,
    )
    found = {e.text: e.label for e in extractor.extract([make_paper("A study", abstract)])[0]}
    assert found["Broad Institute"] is EntityLabel.ORGANIZATION
    assert found["pegRNA"] is EntityLabel.TECHNICAL_TERM


def test_passes_do_not_produce_overlapping_spans() -> None:
    # Overlapping highlights are unrenderable, so the pattern pass must skip
    # anything the model already claimed.
    abstract = "The BRCA1 gene was studied."
    extractor = build_stub_extractor(
        {"A study": [], abstract: [StubEnt("BRCA1", "GENE_OR_GENE_PRODUCT", 4, 9)]},
        scientific=True,
    )
    spans = [
        (span.start, span.end)
        for entity in extractor.extract([make_paper("A study", abstract)])[0]
        for span in entity.spans
        if span.field == "abstract"
    ]
    for i, (start, end) in enumerate(spans):
        for other_start, other_end in spans[i + 1 :]:
            assert start >= other_end or other_start >= end


def test_a_surface_form_gets_exactly_one_label() -> None:
    # "PE" was previously emitted twice for one paper — as a technical term
    # and an organization — which is noise in a filter list.
    abstract = "PE is a method. PE works well. PE again."
    extractor = build_stub_extractor(
        {"A study": [], abstract: [StubEnt("PE", "ORG", 16, 18)]},
        scientific=False,
    )
    entities = extractor.extract([make_paper("A study", abstract)])[0]
    assert [e.text for e in entities].count("PE") == 1
    assert entities[0].label is EntityLabel.TECHNICAL_TERM  # scientific label wins


def test_extractor_falls_back_when_no_model_can_be_loaded() -> None:
    extractor = build_entity_extractor("definitely-not-a-real-spacy-model")
    assert isinstance(extractor, PatternEntityExtractor)
    assert extractor.model_id == "pattern-only"
