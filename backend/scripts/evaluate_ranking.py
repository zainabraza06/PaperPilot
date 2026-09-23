"""Score every ranking strategy against the golden set.

Runs offline against the frozen candidate pool, so the numbers are
reproducible: same pool, same judgments, same model, same result.

What is being measured, precisely: **re-ranking quality within the
retrieved candidate set.** Every candidate is judged, so there is no
pooling bias and no unjudged-document assumption doing hidden work. What
is *not* being measured is the retrieval layer's coverage of the wider
literature — that would need judgments over papers the fan-out never
returned, which this golden set does not have.

The `retrieval-order` row is the baseline that matters: it is the order
the fan-out already produces with no ranking at all. A ranker that cannot
beat it is not earning its dependencies.

With ``--clusters`` it also reports what the topic clusterer does to each
frozen pool. That has no ground truth to score against - nobody hand-labelled
the "correct" sub-topics - so it reports the silhouette score, the group
sizes and the generated labels, and leaves the judgement of whether those
labels are useful to a reader. Reporting an unvalidated number as if it
were an accuracy would be worse than reporting nothing.

Usage::

    python -m scripts.evaluate_ranking
    python -m scripts.evaluate_ranking --sweep-alpha
    python -m scripts.evaluate_ranking --clusters
    python -m scripts.evaluate_ranking --markdown > ../docs/ranking-results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from app.config import get_settings
from app.core.logging import configure_logging
from app.models.paper import Paper
from app.services.enrichment.clustering import TopicClusterer
from app.services.ranking.cache import CachedEmbedder
from app.services.ranking.embeddings import Embedder, build_embedder
from app.services.ranking.evaluation import (
    RankingMetrics,
    average_metrics,
    evaluate_ranking,
)
from app.services.ranking.hybrid import FusionStrategy, HybridRanker

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
KS = (5, 10, 20)


class Run(NamedTuple):
    """One evaluated configuration."""

    label: str
    metrics: RankingMetrics


def load_pool() -> dict[str, list[Paper]]:
    raw = json.loads((EVAL_DIR / "pool.json").read_text(encoding="utf-8"))
    return {
        query_id: [Paper.model_validate(item) for item in items]
        for query_id, items in raw.items()
    }


def load_judgments() -> dict[str, dict[str, int]]:
    raw = json.loads((EVAL_DIR / "judgments.json").read_text(encoding="utf-8"))
    return {
        query_id: {paper_id: entry["grade"] for paper_id, entry in entries.items()}
        for query_id, entries in raw.items()
    }


def load_queries() -> dict[str, str]:
    entries = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    return {entry["id"]: entry["query"] for entry in entries}


def evaluate_strategy(
    ranker: HybridRanker | None,
    pool: dict[str, list[Paper]],
    queries: dict[str, str],
    judgments: dict[str, dict[str, int]],
) -> RankingMetrics:
    """Macro-average one configuration across every query.

    ``ranker=None`` evaluates the pool in retrieval order — the do-nothing
    baseline.
    """
    per_query: list[RankingMetrics] = []
    for query_id, papers in pool.items():
        if ranker is None:
            ranked_ids = [paper.id for paper in papers]
        else:
            ranked_ids = [paper.id for paper in ranker.rank(queries[query_id], papers)]
        per_query.append(evaluate_ranking(ranked_ids, judgments[query_id], KS))
    return average_metrics(per_query, KS)


def build_runs(embedder: Embedder, alpha: float, sweep: bool) -> list[tuple[str, HybridRanker | None]]:
    runs: list[tuple[str, HybridRanker | None]] = [("retrieval-order (no ranking)", None)]
    for strategy in (FusionStrategy.LEXICAL, FusionStrategy.SEMANTIC):
        runs.append((f"{strategy.value} only", HybridRanker(embedder, strategy=strategy)))
    runs.append(
        (
            f"hybrid linear (alpha={alpha})",
            HybridRanker(embedder, strategy=FusionStrategy.LINEAR, alpha=alpha),
        )
    )
    runs.append(("hybrid RRF", HybridRanker(embedder, strategy=FusionStrategy.RRF)))
    # The ablation for the document-length prior. Without this row the
    # prior is an unfalsifiable claim.
    runs.append(
        (
            "  ...minus the evidence prior",
            HybridRanker(
                embedder, strategy=FusionStrategy.LINEAR, alpha=alpha, evidence_k=0.0
            ),
        )
    )
    if sweep:
        for value in (0.2, 0.3, 0.4, 0.5, 0.7, 0.8):
            runs.append(
                (
                    f"hybrid linear (alpha={value})",
                    HybridRanker(embedder, strategy=FusionStrategy.LINEAR, alpha=value),
                )
            )
    return runs


def render_table(runs: Sequence[Run], markdown: bool) -> str:
    headers = (
        ["strategy"]
        + [f"R@{k}" for k in KS]
        + [f"NDCG@{k}" for k in KS]
        + ["MRR"]
    )
    rows = [[run.label, *run.metrics.summary_row(KS)] for run in runs]
    widths = [
        max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        padded = [str(cell).ljust(widths[i]) for i, cell in enumerate(cells)]
        return ("| " + " | ".join(padded) + " |") if markdown else "  ".join(padded)

    out = [line(headers)]
    if markdown:
        out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    else:
        out.append("-" * (sum(widths) + 2 * len(widths)))
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-alpha", action="store_true", help="Also evaluate other alphas")
    parser.add_argument("--markdown", action="store_true", help="Emit a markdown table")
    parser.add_argument("--per-query", action="store_true", help="Break the winner down by query")
    parser.add_argument("--clusters", action="store_true", help="Also report topic clustering")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    configure_logging("WARNING")

    settings = get_settings()
    pool, judgments, queries = load_pool(), load_judgments(), load_queries()
    embedder = CachedEmbedder(build_embedder(settings.embedding_model))

    judged = sum(len(v) for v in judgments.values())
    relevant = sum(1 for v in judgments.values() for g in v.values() if g >= 2)
    print(f"embedder      : {embedder.model_id}")
    print(f"queries       : {len(pool)}")
    print(f"candidates    : {judged} (all judged; {relevant} relevant at grade >= 2)")

    # Recall@k is bounded by min(k, |relevant|) / |relevant|. With 14 relevant
    # papers in a 23-candidate pool, a *perfect* Recall@5 is 0.357, not 1.0.
    # Printing the ceiling stops a correct number reading as a bad one.
    ceilings = evaluate_strategy(None, pool, queries, judgments)
    ceiling_text = ", ".join(
        f"R@{k}<={ceilings.recall_ceiling_at_k[k]:.3f}" for k in KS
    )
    print(f"recall ceiling: {ceiling_text}  (relevant sets are larger than k)")
    print()

    runs = [
        Run(label, evaluate_strategy(ranker, pool, queries, judgments))
        for label, ranker in build_runs(embedder, settings.ranking_alpha, args.sweep_alpha)
    ]
    print(render_table(runs, args.markdown))

    if args.per_query:
        print("\nper-query, hybrid linear:")
        ranker = HybridRanker(
            embedder, strategy=FusionStrategy.LINEAR, alpha=settings.ranking_alpha
        )
        for query_id, papers in pool.items():
            ranked_ids = [p.id for p in ranker.rank(queries[query_id], papers)]
            metrics = evaluate_ranking(ranked_ids, judgments[query_id], KS)
            print(
                f"  {query_id:<24} R@10={metrics.recall_at_k[10]:.3f}"
                f"/{metrics.recall_ceiling_at_k[10]:.3f}"
                f" ({metrics.recall_attainment(10):>5.1%} of ceiling) "
                f"NDCG@10={metrics.ndcg_at_k[10]:.3f} MRR={metrics.mrr:.3f}"
            )

    if args.clusters:
        report_clustering(pool, settings.max_clusters, embedder)
    return 0


def report_clustering(
    pool: dict[str, list[Paper]], max_clusters: int, embedder: Embedder
) -> None:
    """Describe what the clusterer does to each pool.

    Deliberately descriptive, not scored: there is no hand-labelled ground
    truth for sub-topics, so this prints what a reviewer needs to judge the
    output themselves - how many groups, how well separated, how big, and
    what they were named.
    """
    clusterer = TopicClusterer(embedder, max_clusters=max_clusters)
    clustered = 0
    print("\nclustering (ward-agglomerative, silhouette-selected k):")
    for query_id, papers in pool.items():
        result = clusterer.cluster(papers)
        if not result.report.applied:
            print(f"  {query_id:<24} n={len(papers):<3} not split - {result.report.reason}")
            continue
        clustered += 1
        sizes = "/".join(str(cluster.size) for cluster in result.clusters)
        print(
            f"  {query_id:<24} n={len(papers):<3} k={result.report.clusters} "
            f"sizes={sizes:<10} silhouette={result.report.silhouette}"
        )
        for cluster in result.clusters:
            print(f"      [{cluster.size:>2}] {cluster.label}")
    print(f"\n  {clustered}/{len(pool)} pools split into sub-topics")


if __name__ == "__main__":
    sys.exit(main())
