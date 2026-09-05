"""Topic clustering tests.

Run against a stub embedder that places papers at chosen coordinates, so
the geometry is exact and the assertions are about the clusterer's
decisions rather than about a model's behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from app.models.paper import Paper, SourceName
from app.services.enrichment.clustering import TopicClusterer, _c_tf_idf, _format_label
from app.services.ranking.cache import CachedEmbedder
from app.services.ranking.embeddings import HashingEmbedder


class PlacedEmbedder:
    """Places each paper at a coordinate chosen by a marker in its text."""

    dimension = 2
    model_id = "placed"

    def __init__(self, markers: dict[str, tuple[float, float]]) -> None:
        self._markers = markers

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            for marker, point in self._markers.items():
                if marker in text:
                    rows.append(point)
                    break
            else:
                rows.append((0.0, 0.0))
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def make_paper(index: int, title: str, abstract: str) -> Paper:
    return Paper(
        id=f"p{index}",
        source=SourceName.ARXIV,
        source_id=str(index),
        title=title,
        abstract=abstract,
        url=f"https://example.org/{index}",
    )


def two_group_papers(per_group: int = 5) -> list[Paper]:
    """Two well-separated topics, large enough for clustering to engage."""
    papers = []
    for i in range(per_group):
        papers.append(make_paper(i, f"ALPHA genome editing {i}", "crispr pegRNA genome editing"))
    for i in range(per_group):
        papers.append(
            make_paper(per_group + i, f"BETA solar cells {i}", "photovoltaic silicon solar cells")
        )
    return papers


@pytest.fixture
def separated_clusterer() -> TopicClusterer:
    return TopicClusterer(
        PlacedEmbedder({"ALPHA": (1.0, 0.0), "BETA": (0.0, 1.0)}), min_papers=8
    )


def test_two_distinct_topics_are_separated(separated_clusterer: TopicClusterer) -> None:
    result = separated_clusterer.cluster(two_group_papers())
    assert result.report.applied is True
    assert result.report.clusters == 2
    assert sorted(c.size for c in result.clusters) == [5, 5]


def test_every_paper_is_assigned_to_exactly_one_cluster(
    separated_clusterer: TopicClusterer,
) -> None:
    papers = two_group_papers()
    result = separated_clusterer.cluster(papers)
    assigned = [pid for cluster in result.clusters for pid in cluster.paper_ids]
    assert sorted(assigned) == sorted(p.id for p in papers)
    assert len(assigned) == len(set(assigned))


def test_clusters_are_named_from_their_distinctive_terms(
    separated_clusterer: TopicClusterer,
) -> None:
    result = separated_clusterer.cluster(two_group_papers())
    labels = " ".join(c.label for c in result.clusters)
    assert "solar" in labels or "photovoltaic" in labels
    assert "genome" in labels or "crispr" in labels or "pegrna" in labels


def test_clusters_are_ordered_largest_first() -> None:
    papers = two_group_papers(per_group=5)
    papers += [make_paper(90 + i, f"GAMMA quantum {i}", "quantum chromodynamics") for i in range(3)]
    clusterer = TopicClusterer(
        PlacedEmbedder({"ALPHA": (1.0, 0.0), "BETA": (0.0, 1.0), "GAMMA": (-1.0, -1.0)}),
        min_papers=8,
    )
    result = clusterer.cluster(papers)
    sizes = [c.size for c in result.clusters]
    assert sizes == sorted(sizes, reverse=True)


def test_a_small_result_set_is_not_clustered(separated_clusterer: TopicClusterer) -> None:
    result = separated_clusterer.cluster(two_group_papers(per_group=2))
    assert result.report.applied is False
    assert "at least 8" in (result.report.reason or "")
    assert result.clusters == []


def test_an_empty_result_set_is_handled(separated_clusterer: TopicClusterer) -> None:
    assert separated_clusterer.cluster([]).report.applied is False


def test_a_single_coherent_topic_is_reported_rather_than_split_arbitrarily() -> None:
    # Every paper at the same point: no split is meaningful, and inventing
    # one would mislead. This is the diffusion-image case from the eval set.
    papers = [make_paper(i, f"ALPHA topic {i}", "one single coherent subject") for i in range(12)]
    clusterer = TopicClusterer(PlacedEmbedder({"ALPHA": (1.0, 0.0)}), min_papers=8)
    result = clusterer.cluster(papers)
    assert result.report.applied is False
    assert result.clusters == []
    assert "one coherent topic" in (result.report.reason or "")


def test_a_lone_outlier_does_not_become_its_own_cluster() -> None:
    # Silhouette alone prefers this split (it scores very highly), but a
    # cluster of one is not a sub-topic a user can navigate to.
    papers = [make_paper(i, f"ALPHA topic {i}", "shared subject matter") for i in range(11)]
    papers.append(make_paper(99, "BETA unrelated", "completely different subject"))
    clusterer = TopicClusterer(
        PlacedEmbedder({"ALPHA": (1.0, 0.0), "BETA": (0.0, 1.0)}), min_papers=8
    )
    result = clusterer.cluster(papers)
    assert all(cluster.size >= 3 for cluster in result.clusters)


def test_the_report_carries_the_algorithm_and_its_score(
    separated_clusterer: TopicClusterer,
) -> None:
    report = separated_clusterer.cluster(two_group_papers()).report
    assert report.algorithm == "ward-agglomerative"
    assert report.silhouette is not None and report.silhouette > 0
    assert report.clusters == 2


def test_max_clusters_is_respected() -> None:
    papers = [make_paper(i, f"T{i % 8} topic {i}", f"subject {i % 8}") for i in range(40)]
    clusterer = TopicClusterer(CachedEmbedder(HashingEmbedder()), max_clusters=3, min_papers=8)
    result = clusterer.cluster(papers)
    assert result.report.clusters <= 3


def test_clustering_runs_on_a_real_embedder_without_error() -> None:
    # A smoke test over the actual code path, hashing embedder aside.
    result = TopicClusterer(CachedEmbedder(HashingEmbedder()), min_papers=8).cluster(
        two_group_papers()
    )
    assert result.report.applied in (True, False)  # either outcome is valid
    assert isinstance(result.clusters, list)


# --- labelling ------------------------------------------------------------


def test_c_tf_idf_discounts_terms_shared_by_every_cluster() -> None:
    # "editing" is in both clusters, so it must not be what names either of
    # them, even though it is the single most frequent term overall.
    terms = _c_tf_idf(
        {
            0: [["editing", "editing", "pegrna"], ["editing", "pegrna", "prime"]],
            1: [["editing", "editing", "solar"], ["editing", "solar", "silicon"]],
        },
        top_n=2,
    )
    assert terms[0] == ["pegrna", "prime"]
    assert terms[1] == ["solar", "silicon"]


def test_c_tf_idf_prefers_terms_exclusive_to_one_cluster() -> None:
    terms = _c_tf_idf(
        {
            0: [["shared", "shared", "unique"]],
            1: [["shared", "shared", "shared"]],
        },
        top_n=1,
    )
    assert terms[0] == ["unique"]


def test_c_tf_idf_skips_very_short_and_numeric_terms() -> None:
    terms = _c_tf_idf({0: [["ab", "42", "genome"]], 1: [["other"]]}, top_n=3)
    assert terms[0] == ["genome"]


def test_c_tf_idf_on_no_clusters_returns_nothing() -> None:
    assert _c_tf_idf({}, top_n=3) == {}


def test_a_cluster_with_no_distinctive_terms_still_gets_a_name() -> None:
    assert _format_label([]) == "Miscellaneous"


def test_label_joins_the_top_three_terms() -> None:
    assert _format_label(["alpha", "beta", "gamma", "delta"]) == "alpha · beta · gamma"
