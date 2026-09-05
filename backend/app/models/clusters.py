"""Thematic groupings of a result set."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicCluster(BaseModel):
    """One sub-topic within a result set.

    ``terms`` is what makes this usable in a UI: a cluster identified only
    by a number is no more navigable than the flat list it replaced, so
    every cluster carries the terms that distinguish it from its siblings.
    """

    id: int = Field(ge=0)
    label: str = Field(description="Short human-readable name, built from `terms`.")
    terms: list[str] = Field(default_factory=list, description="Distinctive terms, best first.")
    paper_ids: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.paper_ids)


class ClusteringReport(BaseModel):
    """Whether clustering ran, and how it went.

    Mirrors ``RankingReport``: an enrichment that did not happen is a state
    the UI has to render, not an error to swallow. A result set too small
    to cluster is normal, not a failure.
    """

    applied: bool
    algorithm: str | None = None
    clusters: int = 0
    #: Mean silhouette score of the chosen partition, in [-1, 1]. Around 0
    #: means the groups overlap heavily and the split is not meaningful.
    silhouette: float | None = None
    elapsed_ms: int = 0
    reason: str | None = Field(default=None, description="Why clustering did not run.")
