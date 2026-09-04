"""Ranking evaluation metrics.

Standard IR metrics, implemented here rather than pulled in so the exact
definition used is visible and testable — "NDCG@10" means several subtly
different things across libraries, and a portfolio claim about ranking
quality is only as good as the definition behind it.

Relevance judgments are **graded**, on a 0-3 scale:

===  ==========================================================
  3  Directly answers the query; a researcher would definitely read it.
  2  Clearly on-topic and useful.
  1  Marginally related — same field, different question.
  0  Not relevant.
===  ==========================================================

Recall@k uses a binary cut at grade >= 2, because "did the useful papers
make the first page" is the question a researcher actually has. NDCG uses
the full grades, because it is the metric that distinguishes *ordering*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

#: Judgments at or above this grade count as relevant for binary metrics.
RELEVANT_GRADE = 2


class RankingMetrics(BaseModel):
    """Metrics for one query, or averaged across a query set."""

    recall_at_k: dict[int, float] = Field(default_factory=dict)
    recall_ceiling_at_k: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "The best Recall@k any ranking could achieve, min(k, R)/R. When a query "
            "has more relevant papers than k, Recall@k cannot reach 1.0 and reporting "
            "it without this context understates a perfect ranking."
        ),
    )
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    judged_relevant: int = 0
    retrieved: int = 0

    def recall_attainment(self, k: int) -> float:
        """Recall@k as a fraction of the best it could possibly have been.

        This is the number to read when the relevant set is larger than k:
        1.000 means the ranking put every relevant paper it could into the
        top k, even though Recall@k itself is well below 1.
        """
        ceiling = self.recall_ceiling_at_k.get(k, 0.0)
        return self.recall_at_k.get(k, 0.0) / ceiling if ceiling else 0.0

    def summary_row(self, ks: Sequence[int]) -> list[str]:
        """Format as a table row, for the evaluation script's output."""
        cells = [f"{self.recall_at_k.get(k, 0.0):.3f}" for k in ks]
        cells += [f"{self.ndcg_at_k.get(k, 0.0):.3f}" for k in ks]
        cells.append(f"{self.mrr:.3f}")
        return cells


def evaluate_ranking(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, int],
    ks: Sequence[int] = (5, 10, 20),
) -> RankingMetrics:
    """Score one ranked list against graded judgments.

    Args:
        ranked_ids: paper ids in rank order, best first.
        judgments: paper id -> grade. Ids absent from this mapping are
            treated as grade 0. This is the standard "unjudged is
            irrelevant" assumption, and it is only sound because the
            judgment pool was built from the union of every system's
            output — see ``scripts/build_golden_set.py``.
        ks: cutoffs to report.
    """
    relevant_ids = {pid for pid, grade in judgments.items() if grade >= RELEVANT_GRADE}
    grades = [judgments.get(pid, 0) for pid in ranked_ids]

    metrics = RankingMetrics(
        judged_relevant=len(relevant_ids),
        retrieved=len(ranked_ids),
    )

    for k in ks:
        top_k = grades[:k]
        hits = sum(1 for grade in top_k if grade >= RELEVANT_GRADE)
        total_relevant = len(relevant_ids)
        metrics.recall_at_k[k] = hits / total_relevant if total_relevant else 0.0
        metrics.recall_ceiling_at_k[k] = (
            min(k, total_relevant) / total_relevant if total_relevant else 0.0
        )
        metrics.precision_at_k[k] = hits / k if k else 0.0
        metrics.ndcg_at_k[k] = _ndcg(grades, list(judgments.values()), k)

    metrics.mrr = _reciprocal_rank(grades)
    return metrics


def _dcg(grades: Sequence[int], k: int) -> float:
    """Discounted cumulative gain with the exponential gain formulation.

    ``(2**grade - 1) / log2(rank + 1)`` — the standard form, which rewards
    a grade-3 result far more than a grade-1 one rather than linearly.
    """
    return float(
        sum(
            (2**grade - 1) / math.log2(position + 2)
            for position, grade in enumerate(grades[:k])
        )
    )


def _ndcg(grades: Sequence[int], all_grades: Sequence[int], k: int) -> float:
    """DCG normalized by the DCG of the ideal ranking of the judged set."""
    ideal = sorted(all_grades, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(grades, k) / ideal_dcg


def _reciprocal_rank(grades: Sequence[int]) -> float:
    """1 / rank of the first relevant result, or 0 if there is none."""
    for position, grade in enumerate(grades, start=1):
        if grade >= RELEVANT_GRADE:
            return 1.0 / position
    return 0.0


def average_metrics(per_query: Sequence[RankingMetrics], ks: Sequence[int]) -> RankingMetrics:
    """Macro-average across queries.

    Macro rather than micro: every query counts equally, so a single query
    with many relevant papers cannot dominate the headline number.
    """
    if not per_query:
        return RankingMetrics()
    count = len(per_query)
    return RankingMetrics(
        recall_at_k={k: sum(m.recall_at_k.get(k, 0.0) for m in per_query) / count for k in ks},
        recall_ceiling_at_k={
            k: sum(m.recall_ceiling_at_k.get(k, 0.0) for m in per_query) / count for k in ks
        },
        precision_at_k={
            k: sum(m.precision_at_k.get(k, 0.0) for m in per_query) / count for k in ks
        },
        ndcg_at_k={k: sum(m.ndcg_at_k.get(k, 0.0) for m in per_query) / count for k in ks},
        mrr=sum(m.mrr for m in per_query) / count,
        judged_relevant=sum(m.judged_relevant for m in per_query),
        retrieved=sum(m.retrieved for m in per_query),
    )
