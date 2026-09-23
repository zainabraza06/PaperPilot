"""Tests for the reference-free quality metrics.

Each metric is checked against a case where its value is known by
construction — an exact copy, a pure rewrite, a summary drawn only from
the opening — rather than against a recorded number from a model. A test
that asserts "coverage is 0.62" would only be testing that the embedder
has not changed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from app.services.summarization.quality import (
    QualityScorer,
    QualityScores,
    _lead_bias,
    _longest_copied_span,
    _novelty,
    _tokenize,
)

ABSTRACT = (
    "Antimicrobial resistance is spreading between bacterial species faster "
    "than new antibiotics are being developed. We sequenced two hundred "
    "clinical isolates collected over three years at a single hospital. "
    "Conjugative plasmids carried the majority of the resistance genes we "
    "found. Resistance transfer was observed in every ward we sampled. "
    "We conclude that plasmid surveillance should accompany routine "
    "susceptibility testing in hospital infection control programmes."
)


class StubEmbedder:
    """Embeds by content-word overlap, so similarity is exactly predictable.

    A real sentence-transformer would make these assertions depend on a
    downloaded model's behaviour, which is not what is under test here.
    """

    dimension = 64
    model_id = "stub"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in _tokenize(text):
                vector[hash(token) % self.dimension] += 1.0
            rows.append(vector)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


@pytest.fixture
def scorer() -> QualityScorer:
    return QualityScorer(StubEmbedder())


# --- novelty and copying ------------------------------------------------


def test_an_exact_copy_has_no_novelty_and_is_copied_end_to_end(
    scorer: QualityScorer,
) -> None:
    scores = scorer.score(ABSTRACT, ABSTRACT)
    assert scores is not None
    assert scores.novelty == pytest.approx(0.0)
    assert scores.longest_copied_span == pytest.approx(1.0)
    assert scores.compression == pytest.approx(1.0)


def test_novelty_counts_recombined_phrases_not_reused_vocabulary() -> None:
    """Rewriting reuses the nouns; it is the *pairings* that are new.

    This is why novelty is measured over bigrams. Every content word below
    appears in the source, so unigram novelty would score this 0 and call
    a genuine rewrite a copy.
    """
    abstract = _tokenize("plasmids carried the resistance genes we found")
    rewrite = _tokenize("the genes we found were carried by plasmids")
    assert _novelty(rewrite, abstract) > 0.5


def test_a_long_lifted_clause_is_caught_even_when_novelty_looks_healthy() -> None:
    """The case ``novelty`` alone misses.

    A summary can rewrite most of itself and still lift one clause whole.
    Bigram novelty stays respectable while a long verbatim run sits inside
    it, so the two metrics are reported together.
    """
    abstract = _tokenize(ABSTRACT)
    lifted = _tokenize(
        "The authors report that conjugative plasmids carried the majority "
        "of the resistance genes we found, which is notable."
    )
    assert _novelty(lifted, abstract) > 0.3
    assert _longest_copied_span(lifted, abstract) > 0.4


def test_nothing_in_common_means_no_copied_span() -> None:
    assert _longest_copied_span(_tokenize("entirely unrelated wording"), _tokenize(ABSTRACT)) == 0.0


# --- lead bias ----------------------------------------------------------


def test_a_summary_of_only_the_opening_scores_as_lead_biased(
    scorer: QualityScorer,
) -> None:
    """The failure mode this metric exists to name.

    A model that rewrites the first two sentences has produced fluent,
    grounded text that omits the finding entirely — and every grounding
    check passes it.
    """
    opening = (
        "Antimicrobial resistance is spreading between bacterial species faster "
        "than new antibiotics are being developed. We sequenced two hundred "
        "clinical isolates collected over three years at a single hospital."
    )
    scores = scorer.score(opening, ABSTRACT)
    assert scores is not None
    assert scores.lead_bias < 0.3


def test_a_summary_spanning_the_abstract_is_not_lead_biased(
    scorer: QualityScorer,
) -> None:
    spanning = (
        "Antimicrobial resistance is spreading between bacterial species. "
        "We conclude that plasmid surveillance should accompany routine "
        "susceptibility testing in hospital infection control programmes."
    )
    scores = scorer.score(spanning, ABSTRACT)
    assert scores is not None
    assert scores.lead_bias > 0.4


def test_a_single_source_sentence_has_no_position_to_report() -> None:
    """One row means there is no ordering; 0.5 says "neutral", not "front"."""
    assert _lead_bias(np.array([[1.0, 0.2]])) == 0.5


# --- coverage and redundancy --------------------------------------------


def test_covering_more_of_the_abstract_raises_coverage(scorer: QualityScorer) -> None:
    narrow = "We sequenced two hundred clinical isolates at a single hospital."
    broad = (
        "We sequenced two hundred clinical isolates at a single hospital. "
        "Conjugative plasmids carried most of the resistance genes. "
        "Transfer was seen in every ward, so plasmid surveillance should "
        "accompany routine susceptibility testing."
    )
    narrow_scores = scorer.score(narrow, ABSTRACT)
    broad_scores = scorer.score(broad, ABSTRACT)
    assert narrow_scores is not None and broad_scores is not None
    assert broad_scores.coverage > narrow_scores.coverage


def test_repeating_a_sentence_is_reported_as_redundancy(scorer: QualityScorer) -> None:
    repeated = (
        "Conjugative plasmids carried the majority of the resistance genes. "
        "The majority of the resistance genes were carried by conjugative plasmids."
    )
    scores = scorer.score(repeated, ABSTRACT)
    assert scores is not None
    assert scores.redundancy > 0.9


def test_a_one_sentence_summary_cannot_be_redundant(scorer: QualityScorer) -> None:
    scores = scorer.score("Conjugative plasmids carried the resistance genes.", ABSTRACT)
    assert scores is not None
    assert scores.redundancy == 0.0


# --- unscorable input ---------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "abstract"),
    [
        ("", ABSTRACT),
        (ABSTRACT, ""),
        ("Too short.", ABSTRACT),
        ("   ", "   "),
    ],
)
def test_unscorable_pairs_return_none_rather_than_zeros(
    scorer: QualityScorer, summary: str, abstract: str
) -> None:
    """Zeros would be averaged in as if the summary had scored badly.

    Returning ``None`` forces the caller to exclude the pair instead,
    which is the difference between "we could not measure this" and "this
    measured zero".
    """
    assert scorer.score(summary, abstract) is None


def test_every_reported_field_is_present_in_the_dict(scorer: QualityScorer) -> None:
    scores = scorer.score(ABSTRACT, ABSTRACT)
    assert scores is not None
    assert set(scores.as_dict()) == set(QualityScores.fields())
