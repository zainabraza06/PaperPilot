"""Grouping a result set into readable sub-topics.

Fifty papers in one flat list is the problem this solves: a researcher
scanning results for "diffusion models" wants to see that the set splits
into text-to-image generation, medical imaging, and detection — and to
click one.

Two decisions carry most of the weight:

**How many clusters, and which linkage.** Both were measured rather than
assumed. Maximizing silhouette alone is actively wrong here: it prefers
isolating outliers, and on a 24-paper "diffusion models" result set the
best-scoring partition was 23 papers plus one singleton (silhouette 0.481)
— a perfect score for a split that gives a user nothing to click.

Average linkage on cosine distance produced that singleton behaviour on
every query tested. Ward linkage minimizes within-cluster variance and
produced balanced partitions instead ([15, 7], [18, 6], [16, 4, 3]), so
Ward is what runs. It needs Euclidean distance, which is legitimate here
because the embedder returns L2-normalized vectors and squared Euclidean
distance is then ``2(1 - cosine)`` — the same geometry, monotonically.

Two constraints then encode "is this partition *useful*", which silhouette
does not measure: no cluster below a floor size, and no cluster holding
more than 80% of the results. A partition failing both at every *k* means
the result set is one coherent topic, and saying so is more honest than
presenting an arbitrary split as meaningful.

**How they are named.** A cluster labelled "Cluster 3" is no more
navigable than the flat list it replaced. Labels come from c-TF-IDF: term
frequency within a cluster, discounted by how many clusters the term
appears in. That discount is what stops every cluster in a "prime editing"
search from being labelled "prime editing" — the shared terms cancel and
the distinguishing ones surface.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence

import numpy as np

from app.core.logging import get_logger
from app.models.clusters import ClusteringReport, TopicCluster
from app.models.paper import Paper
from app.services.ranking.document import document_text, tokenize
from app.services.ranking.embeddings import Embedder

logger = get_logger(__name__)

#: Below this, clustering is noise: three groups of two papers tells a
#: researcher nothing they could not see by reading the six titles.
_MIN_PAPERS = 8

#: Each cluster should hold at least this many papers on average, which
#: bounds how finely a small result set may be split.
_MIN_PAPERS_PER_CLUSTER = 3

#: A partition scoring below this is reported but not applied: the groups
#: overlap so heavily that splitting them would mislead rather than help.
_MIN_SILHOUETTE = 0.02

#: A cluster holding more of the result set than this leaves the user with
#: what they already had - a flat list - plus a tab to click first.
_MAX_DOMINANCE = 0.8

#: Reported so a UI (and a reader) knows what produced the grouping.
_ALGORITHM = "ward-agglomerative"

#: Terms shorter than this are too generic to name a topic with.
_MIN_TERM_LENGTH = 3


class ClusterResult:
    """Clusters plus the report describing how they were produced."""

    def __init__(
        self,
        clusters: list[TopicCluster],
        report: ClusteringReport,
        assignments: dict[str, int] | None = None,
    ) -> None:
        self.clusters = clusters
        self.report = report
        self.assignments = assignments or {}


class TopicClusterer:
    """Groups a result set by embedding similarity and names the groups."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        max_clusters: int = 6,
        min_papers: int = _MIN_PAPERS,
        terms_per_cluster: int = 4,
    ) -> None:
        self._embedder = embedder
        self._max_clusters = max_clusters
        self._min_papers = min_papers
        self._terms_per_cluster = terms_per_cluster

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    def cluster(self, papers: Sequence[Paper]) -> ClusterResult:
        """Partition ``papers`` into labelled sub-topics.

        Never raises for lack of data: too few papers, or a partition too
        weak to be meaningful, come back as a report with ``applied=False``
        and a reason the UI can show.
        """
        started = time.perf_counter()

        if len(papers) < self._min_papers:
            return ClusterResult(
                [],
                ClusteringReport(
                    applied=False,
                    reason=f"only {len(papers)} papers; clustering needs at least "
                    f"{self._min_papers} to be meaningful",
                ),
            )

        vectors = self._embedder.encode([document_text(paper) for paper in papers])
        labels, k, silhouette = self._partition(vectors)

        if labels is None:
            return ClusterResult(
                [],
                ClusteringReport(
                    applied=False,
                    algorithm=_ALGORITHM,
                    silhouette=round(float(silhouette), 4) if silhouette is not None else None,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                    reason="these results form one coherent topic; no split "
                    "produced groups that were both distinct and useful",
                ),
            )

        clusters = self._build_clusters(papers, labels, k)
        assignments = {
            paper.id: int(label) for paper, label in zip(papers, labels, strict=True)
        }
        return ClusterResult(
            clusters,
            ClusteringReport(
                applied=True,
                algorithm=_ALGORITHM,
                clusters=len(clusters),
                silhouette=round(float(silhouette), 4) if silhouette is not None else None,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            ),
            assignments,
        )

    # --- partitioning ------------------------------------------------------

    def _partition(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray | None, int, float | None]:
        """Choose k over the *usable* partitions and return the best one.

        Returns ``(None, k, score)`` when no k produced a partition worth
        showing, so the caller can distinguish "did not cluster" from
        "failed to cluster".
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        count = vectors.shape[0]
        highest_k = min(self._max_clusters, count // _MIN_PAPERS_PER_CLUSTER)
        if highest_k < 2:
            return None, 0, None

        best_labels: np.ndarray | None = None
        best_score = -1.0
        best_k = 0
        best_rejected = -1.0  # best score among partitions we declined to use

        for k in range(2, highest_k + 1):
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
                vectors
            )
            sizes = Counter(labels.tolist())
            if len(sizes) < 2:
                continue
            score = float(silhouette_score(vectors, labels, metric="cosine"))
            usable = (
                min(sizes.values()) >= _MIN_PAPERS_PER_CLUSTER
                and max(sizes.values()) / count <= _MAX_DOMINANCE
            )
            logger.debug(
                "k=%d silhouette=%.4f sizes=%s usable=%s",
                k,
                score,
                sorted(sizes.values(), reverse=True),
                usable,
            )
            if not usable:
                best_rejected = max(best_rejected, score)
                continue
            if score > best_score:
                best_labels, best_score, best_k = labels, score, k

        if best_labels is None:
            return None, 0, best_rejected if best_rejected >= 0 else None
        if best_score < _MIN_SILHOUETTE:
            return None, best_k, best_score
        return best_labels, best_k, best_score

    # --- labelling ---------------------------------------------------------

    def _build_clusters(
        self, papers: Sequence[Paper], labels: np.ndarray, k: int
    ) -> list[TopicCluster]:
        members: dict[int, list[Paper]] = {}
        for paper, label in zip(papers, labels, strict=True):
            members.setdefault(int(label), []).append(paper)

        term_lists = _c_tf_idf(
            {
                label: [tokenize(document_text(paper)) for paper in group]
                for label, group in members.items()
            },
            top_n=self._terms_per_cluster,
        )

        clusters = [
            TopicCluster(
                id=label,
                label=_format_label(term_lists.get(label, [])),
                terms=term_lists.get(label, []),
                paper_ids=[paper.id for paper in group],
            )
            for label, group in sorted(members.items())
        ]
        # Largest first: the dominant sub-topic is the one a user most
        # likely wants, and stable ordering keeps tab positions predictable.
        clusters.sort(key=lambda c: (-c.size, c.id))
        return clusters


def _c_tf_idf(
    tokens_by_cluster: dict[int, list[list[str]]], top_n: int
) -> dict[int, list[str]]:
    """Class-based TF-IDF: what makes each cluster different from the others.

    Three factors, each earning its place:

    * **tf** - frequency within the cluster, treated as one document.
    * **idf** - ``log(1 + clusters / clusters_containing_term)``, so a term
      present in every cluster is discounted toward nothing. This is what
      stops all three clusters of a "prime editing" search from being
      labelled "prime editing".
    * **exclusivity** - the share of the term's *total* occurrences falling
      in this cluster. A measured addition: without it the smaller clusters
      were labelled in the parent topic's vocabulary
      ("alphafold - protein - prediction"); with it they name themselves
      ("alphafold - alphafold-multimer - differentiable"). The largest
      cluster stays generic either way, which is correct: it *is* the
      generic topic.
    """
    if not tokens_by_cluster:
        return {}

    counts: dict[int, Counter[str]] = {
        label: Counter(token for document in documents for token in document)
        for label, documents in tokens_by_cluster.items()
    }
    lengths = {label: max(sum(counter.values()), 1) for label, counter in counts.items()}

    total_frequency: Counter[str] = Counter()
    cluster_frequency: Counter[str] = Counter()
    for counter in counts.values():
        total_frequency.update(counter)
        cluster_frequency.update(counter.keys())

    cluster_count = len(counts)
    results: dict[int, list[str]] = {}
    for label, counter in counts.items():
        scored: list[tuple[float, str]] = []
        for term, count in counter.items():
            if len(term) < _MIN_TERM_LENGTH or term.isdigit():
                continue
            tf = count / lengths[label]
            idf = math.log(1 + cluster_count / cluster_frequency[term])
            exclusivity = count / total_frequency[term]
            scored.append((tf * idf * exclusivity, term))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        results[label] = [term for _, term in scored[:top_n]]
    return results


def _format_label(terms: Sequence[str]) -> str:
    """Build a short display name from a cluster's distinctive terms."""
    if not terms:
        return "Miscellaneous"
    return " · ".join(terms[:3])
