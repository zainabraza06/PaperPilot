"""Tests for the IR metrics.

These carry more weight than usual: the README publishes numbers produced
by this module, so a silent bug here would turn into a false public claim
about ranking quality. Every expected value below is hand-computed from
the definition rather than captured from a previous run.
"""

from __future__ import annotations

import math

import pytest

from app.services.ranking.evaluation import (
    RankingMetrics,
    average_metrics,
    evaluate_ranking,
)

# a=3 (highly relevant), c=2 (relevant), b/d=0. Relevant set = {a, c}.
JUDGMENTS = {"a": 3, "b": 0, "c": 2, "d": 0}


def test_recall_counts_only_grade_two_and_above() -> None:
    metrics = evaluate_ranking(["a", "b", "c", "d"], JUDGMENTS, ks=(1, 2, 3, 4))
    assert metrics.recall_at_k[1] == pytest.approx(0.5)  # a of {a, c}
    assert metrics.recall_at_k[2] == pytest.approx(0.5)  # b is grade 0
    assert metrics.recall_at_k[3] == pytest.approx(1.0)
    assert metrics.recall_at_k[4] == pytest.approx(1.0)


def test_precision_divides_by_k_not_by_hits() -> None:
    metrics = evaluate_ranking(["a", "b", "c", "d"], JUDGMENTS, ks=(2, 4))
    assert metrics.precision_at_k[2] == pytest.approx(0.5)
    assert metrics.precision_at_k[4] == pytest.approx(0.5)


def test_recall_ceiling_reflects_that_k_can_be_smaller_than_the_relevant_set() -> None:
    # Two relevant papers, k=1: the best possible Recall@1 is 0.5, not 1.0.
    metrics = evaluate_ranking(["a", "c"], JUDGMENTS, ks=(1, 2))
    assert metrics.recall_ceiling_at_k[1] == pytest.approx(0.5)
    assert metrics.recall_ceiling_at_k[2] == pytest.approx(1.0)


def test_attainment_reports_a_perfect_ranking_as_perfect() -> None:
    # Recall@1 of 0.5 is the maximum achievable, so attainment is 1.0.
    metrics = evaluate_ranking(["a", "c", "b", "d"], JUDGMENTS, ks=(1,))
    assert metrics.recall_at_k[1] == pytest.approx(0.5)
    assert metrics.recall_attainment(1) == pytest.approx(1.0)


def test_ndcg_matches_the_hand_computed_value() -> None:
    # DCG@2 for [3, 0] = (2^3-1)/log2(2) + (2^0-1)/log2(3) = 7.0
    # Ideal order is [3, 2]: IDCG@2 = 7/1 + (2^2-1)/log2(3) = 7 + 3/1.58496
    metrics = evaluate_ranking(["a", "b", "c", "d"], JUDGMENTS, ks=(2,))
    expected = 7.0 / (7.0 + 3.0 / math.log2(3))
    assert metrics.ndcg_at_k[2] == pytest.approx(expected, rel=1e-6)


def test_ndcg_is_one_for_the_ideal_ordering() -> None:
    metrics = evaluate_ranking(["a", "c", "b", "d"], JUDGMENTS, ks=(2, 4))
    assert metrics.ndcg_at_k[2] == pytest.approx(1.0)
    assert metrics.ndcg_at_k[4] == pytest.approx(1.0)


def test_ndcg_uses_graded_relevance_not_a_binary_cut() -> None:
    # Swapping the grade-3 and grade-2 papers must cost something; with a
    # binary notion of relevance these two orderings would score the same.
    ideal = evaluate_ranking(["a", "c"], JUDGMENTS, ks=(2,)).ndcg_at_k[2]
    swapped = evaluate_ranking(["c", "a"], JUDGMENTS, ks=(2,)).ndcg_at_k[2]
    assert swapped < ideal


def test_mrr_is_the_reciprocal_rank_of_the_first_relevant_result() -> None:
    assert evaluate_ranking(["a", "b"], JUDGMENTS).mrr == pytest.approx(1.0)
    assert evaluate_ranking(["b", "a"], JUDGMENTS).mrr == pytest.approx(0.5)
    assert evaluate_ranking(["b", "d", "c"], JUDGMENTS).mrr == pytest.approx(1 / 3)


def test_mrr_is_zero_when_nothing_relevant_was_retrieved() -> None:
    assert evaluate_ranking(["b", "d"], JUDGMENTS).mrr == 0.0


def test_unjudged_documents_are_treated_as_irrelevant() -> None:
    metrics = evaluate_ranking(["unseen", "a"], JUDGMENTS, ks=(1, 2))
    assert metrics.recall_at_k[1] == 0.0
    assert metrics.recall_at_k[2] == pytest.approx(0.5)


def test_binary_metrics_cut_at_grade_two_while_ndcg_uses_every_grade() -> None:
    # Nothing here reaches grade 2, so recall and its ceiling are zero. NDCG
    # is still meaningful: putting the grade-1 paper above the grade-0 one is
    # a better ordering, and the graded metric is supposed to say so.
    metrics = evaluate_ranking(["d", "b"], {"b": 0, "d": 1}, ks=(5,))
    assert metrics.recall_at_k[5] == 0.0
    assert metrics.recall_ceiling_at_k[5] == 0.0
    assert metrics.recall_attainment(5) == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg_at_k[5] == pytest.approx(1.0)  # this *is* the ideal order


def test_ndcg_is_zero_only_when_no_document_has_any_grade() -> None:
    metrics = evaluate_ranking(["b", "d"], {"b": 0, "d": 0}, ks=(5,))
    assert metrics.ndcg_at_k[5] == 0.0


def test_empty_ranking_scores_zero() -> None:
    metrics = evaluate_ranking([], JUDGMENTS, ks=(5,))
    assert metrics.recall_at_k[5] == 0.0
    assert metrics.mrr == 0.0


def test_averaging_is_macro_so_one_big_query_cannot_dominate() -> None:
    # Query one: 1 relevant, found. Query two: 100 relevant, none found.
    # A micro-average would report ~0.01; the macro-average reports 0.5.
    good = evaluate_ranking(["a"], {"a": 3}, ks=(5,))
    bad = evaluate_ranking(["x"], {f"r{i}": 3 for i in range(100)}, ks=(5,))
    averaged = average_metrics([good, bad], ks=(5,))
    assert averaged.recall_at_k[5] == pytest.approx(0.5)


def test_averaging_no_queries_returns_zeroed_metrics() -> None:
    assert average_metrics([], ks=(5,)) == RankingMetrics()
