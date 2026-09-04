"""Tests for the ranking pipeline: document view, BM25, embedders, fusion.

Everything here runs offline against ``HashingEmbedder`` or a stub, so the
suite stays fast and deterministic. The quality of the *real* model is not
what these assert — that is what the golden-set evaluation measures.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from app.models.paper import Paper, SourceName
from app.services.ranking.document import document_text, tokenize
from app.services.ranking.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from app.services.ranking.hybrid import FusionStrategy, HybridRanker
from app.services.ranking.lexical import BM25Scorer


def make_paper(
    title: str,
    abstract: str | None = None,
    *,
    paper_id: str | None = None,
    keywords: list[str] | None = None,
    journal: str | None = None,
) -> Paper:
    return Paper(
        id=paper_id or f"test:{abs(hash(title))}",
        source=SourceName.ARXIV,
        source_id="1",
        title=title,
        abstract=abstract,
        keywords=keywords or [],
        journal=journal,
        url="https://example.org/1",
    )


# --- document view ------------------------------------------------------


def test_title_is_repeated_to_weight_it_above_the_abstract() -> None:
    text = document_text(make_paper("CRISPR", "an abstract"))
    assert text.lower().count("crispr") == 2


def test_papers_without_an_abstract_fall_back_to_keywords_and_venue() -> None:
    paper = make_paper("A title", None, keywords=["genomics"], journal="Nature")
    text = document_text(paper)
    assert "genomics" in text
    assert "Nature" in text


def test_tokenize_drops_stopwords_and_single_characters() -> None:
    tokens = tokenize("The effect of a drug on X cells")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "x" not in tokens
    assert "cells" in tokens


def test_bigrams_are_opt_in_and_pair_adjacent_tokens() -> None:
    assert "prime_editing" not in tokenize("prime editing")
    assert "prime_editing" in tokenize("prime editing", bigrams=True)


# --- embedders -----------------------------------------------------------


def test_hashing_embedder_output_is_unit_length() -> None:
    vectors = HashingEmbedder().encode(["genome editing", "graph neural networks"])
    assert vectors.shape == (2, 256)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder()
    np.testing.assert_array_equal(embedder.encode(["abc"]), embedder.encode(["abc"]))


def test_hashing_embedder_scores_overlap_above_unrelated_text() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.encode(
        ["prime editing genome", "prime editing genomes", "quantum chromodynamics"]
    )
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


def test_empty_text_embeds_without_dividing_by_zero() -> None:
    vectors = HashingEmbedder().encode(["", "a"])
    assert np.isfinite(vectors).all()


def test_build_embedder_falls_back_when_the_model_cannot_load() -> None:
    # The degradation has to be silent to the caller but visible in model_id,
    # so a quality number can never be misattributed to the real model.
    embedder = build_embedder("definitely-not-a-real-model-name-xyz")
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.model_id.startswith("hashing-")


def test_build_embedder_with_no_model_configured_uses_the_fallback() -> None:
    assert isinstance(build_embedder(None), HashingEmbedder)


def test_sentence_transformer_embedder_defers_loading_until_used() -> None:
    # Constructing must be free; the API should start without paying for torch.
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    assert embedder._model is None
    assert embedder.model_id == "all-MiniLM-L6-v2"


# --- BM25 ----------------------------------------------------------------


def test_bm25_ranks_the_matching_paper_first() -> None:
    papers = [
        make_paper("Solar cell efficiency", "photovoltaics silicon"),
        make_paper("Perovskite absorbers", "thin film deposition"),
        make_paper("Antenna arrays", "beamforming design"),
        make_paper("Prime editing efficiency", "genome editing in human cells"),
    ]
    scores = BM25Scorer(papers).score("prime editing")
    assert scores.argmax() == 3


def test_bm25_signal_collapses_on_a_two_document_corpus() -> None:
    """Documents a real limitation rather than papering over it.

    Okapi IDF is zero for a term appearing in half the documents, so with
    two candidates every score is zero regardless of the query. The hybrid
    degrades to the semantic signal there, which is the correct outcome,
    but it must not be mistaken for "the lexical signal disagreed".
    """
    papers = [
        make_paper("Solar cell efficiency", "photovoltaics"),
        make_paper("Prime editing efficiency", "genome editing in human cells"),
    ]
    np.testing.assert_array_equal(BM25Scorer(papers).score("prime editing"), np.zeros(2))


def test_bm25_rewards_rare_terms_over_common_ones() -> None:
    # "editing" appears in every document, so IDF should discount it and let
    # the rare term decide. This is the property the hybrid relies on.
    papers = [
        make_paper("Gene editing review", "editing editing editing"),
        make_paper("Editing with pegRNA", "editing pegRNA"),
        make_paper("Editing outcomes", "editing editing"),
    ]
    scores = BM25Scorer(papers).score("pegRNA editing")
    assert scores.argmax() == 1


def test_bm25_on_an_empty_candidate_set_returns_no_scores() -> None:
    assert BM25Scorer([]).score("anything").size == 0


def test_bm25_survives_papers_with_no_usable_text() -> None:
    # rank_bm25 divides by average document length and raises on an
    # all-empty corpus, so this path has to be guarded.
    papers = [make_paper("a"), make_paper("of")]
    scores = BM25Scorer(papers).score("query")
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_bm25_with_an_all_stopword_query_returns_zeros() -> None:
    papers = [make_paper("Prime editing", "genome")]
    np.testing.assert_array_equal(BM25Scorer(papers).score("the of and"), np.zeros(1))


# --- fusion --------------------------------------------------------------


class StubEmbedder:
    """Returns embeddings from a lookup, so cosine values are exact."""

    dimension = 2
    model_id = "stub"

    def __init__(self, mapping: dict[str, tuple[float, float]]) -> None:
        self._mapping = mapping

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            for key, vector in self._mapping.items():
                if key in text:
                    rows.append(vector)
                    break
            else:
                rows.append((0.0, 0.0))
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


@pytest.fixture
def stub_ranker_papers() -> list[Paper]:
    return [
        make_paper("ALPHA topic", "alpha alpha", paper_id="p-alpha"),
        make_paper("BETA topic", "beta beta", paper_id="p-beta"),
    ]


def test_ranking_populates_a_score_on_every_paper(stub_ranker_papers: list[Paper]) -> None:
    ranker = HybridRanker(HashingEmbedder())
    ranked = ranker.rank("alpha", stub_ranker_papers)
    assert all(paper.score is not None for paper in ranked)
    assert [paper.score.rank for paper in ranked] == [1, 2]  # type: ignore[union-attr]


def test_ranking_does_not_mutate_the_input(stub_ranker_papers: list[Paper]) -> None:
    HybridRanker(HashingEmbedder()).rank("alpha", stub_ranker_papers)
    assert all(paper.score is None for paper in stub_ranker_papers)


def test_combined_score_is_clamped_to_the_zero_one_range() -> None:
    papers = [make_paper(f"paper {n}", "text " * n) for n in range(1, 6)]
    ranked = HybridRanker(HashingEmbedder()).rank("paper text", papers)
    assert all(0.0 <= paper.score.combined <= 1.0 for paper in ranked)  # type: ignore[union-attr]


def test_score_breakdown_keeps_the_raw_components() -> None:
    # The UI shows a breakdown, and an evaluation run needs to attribute a
    # change to the signal that caused it, so the raw values must survive.
    papers = [make_paper("prime editing", "genome editing in human cells")]
    ranked = HybridRanker(HashingEmbedder()).rank("prime editing", papers)
    score = ranked[0].score
    assert score is not None
    assert score.lexical != 0.0 or score.semantic != 0.0
    assert score.strategy == "linear"


def test_semantic_strategy_ignores_the_lexical_signal() -> None:
    embedder = StubEmbedder({"ALPHA": (1.0, 0.0), "BETA": (0.0, 1.0), "query": (0.0, 1.0)})
    papers = [
        make_paper("ALPHA topic", "query query query", paper_id="lexical-match"),
        make_paper("BETA topic", "unrelated words", paper_id="semantic-match"),
    ]
    ranked = HybridRanker(embedder, strategy=FusionStrategy.SEMANTIC).rank("query", papers)
    assert ranked[0].id == "semantic-match"


def test_lexical_strategy_ignores_the_semantic_signal() -> None:
    embedder = StubEmbedder({"ALPHA": (1.0, 0.0), "BETA": (0.0, 1.0), "query": (0.0, 1.0)})
    papers = [
        make_paper("ALPHA topic", "query query query", paper_id="lexical-match"),
        make_paper("BETA topic", "unrelated words", paper_id="semantic-match"),
    ]
    ranked = HybridRanker(embedder, strategy=FusionStrategy.LEXICAL).rank("query", papers)
    assert ranked[0].id == "lexical-match"


def test_alpha_shifts_the_hybrid_between_its_two_signals() -> None:
    embedder = StubEmbedder({"ALPHA": (1.0, 0.0), "BETA": (0.0, 1.0), "query": (0.0, 1.0)})
    papers = [
        make_paper("ALPHA topic", "query query query", paper_id="lexical-match"),
        make_paper("BETA topic", "unrelated words", paper_id="semantic-match"),
    ]
    lexical_heavy = HybridRanker(embedder, alpha=0.0).rank("query", papers)
    semantic_heavy = HybridRanker(embedder, alpha=1.0).rank("query", papers)
    assert lexical_heavy[0].id == "lexical-match"
    assert semantic_heavy[0].id == "semantic-match"


def test_alpha_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="alpha"):
        HybridRanker(HashingEmbedder(), alpha=1.5)


def test_rrf_promotes_a_paper_both_signals_rate_well() -> None:
    # RRF's whole point: a consistent second place beats a first-plus-last.
    embedder = StubEmbedder(
        {"TOP": (1.0, 0.0), "MIDDLE": (0.92, 0.39), "LOW": (0.0, 1.0), "query": (1.0, 0.0)}
    )
    papers = [
        make_paper("TOP paper", "nothing in common here", paper_id="semantic-only"),
        make_paper("MIDDLE paper", "query query", paper_id="consistent"),
        make_paper("LOW paper", "query", paper_id="lexical-only"),
    ]
    ranked = HybridRanker(embedder, strategy=FusionStrategy.RRF).rank("query", papers)
    assert ranked[0].id == "consistent"


def test_ranking_an_empty_set_returns_an_empty_list() -> None:
    assert HybridRanker(HashingEmbedder()).rank("query", []) == []


def test_a_single_candidate_ranks_without_dividing_by_zero() -> None:
    ranked = HybridRanker(HashingEmbedder()).rank("query", [make_paper("only", "one")])
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_identical_candidates_keep_a_stable_order() -> None:
    # With no discriminating signal the ranker must not invent an ordering.
    papers = [make_paper("same title", "same text", paper_id=f"p{n}") for n in range(4)]
    ranked = HybridRanker(HashingEmbedder()).rank("same", papers)
    assert [p.id for p in ranked] == ["p0", "p1", "p2", "p3"]


async def test_arank_matches_rank() -> None:
    papers = [make_paper("prime editing", "genome"), make_paper("solar cells", "silicon")]
    ranker = HybridRanker(HashingEmbedder())
    assert [p.id for p in await ranker.arank("prime editing", papers)] == [
        p.id for p in ranker.rank("prime editing", papers)
    ]
