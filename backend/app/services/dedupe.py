"""Cross-source deduplication.

The same work legitimately appears in all three sources: an arXiv preprint
gets a DOI on publication, Crossref indexes the published version, and
PubMed indexes it again with curated metadata. Showing a researcher the
same paper three times is the single most obvious failure mode of a
multi-source search, so this runs on every result set.

Matching happens in two passes:

1. **DOI match** — exact, after normalization. This is authoritative.
2. **Title match** — for the many records with no DOI (most arXiv
   preprints), on a normalized title key, with a fuzzy fallback for the
   near-misses that punctuation folding does not catch.

Merging is field-wise rather than winner-takes-all: the most complete
record wins the identity, but any field it lacks is filled from a sibling.
That is how a Crossref record with no abstract ends up carrying the arXiv
abstract, which is exactly what the ranking stage needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.core.logging import get_logger
from app.models.paper import Paper, SourceName

logger = get_logger(__name__)

# Above this ratio two normalized titles are the same work. Set high: a
# false merge hides a paper entirely, which is worse than a visible dupe.
_TITLE_SIMILARITY_THRESHOLD = 0.94

# Only titles sharing this prefix are compared pairwise, which keeps the
# fuzzy pass linear in practice instead of quadratic over the whole set.
_BLOCK_PREFIX_LENGTH = 12

# Tie-break when two records are equally complete: curated metadata first,
# then registered metadata, then preprints.
_SOURCE_PRIORITY = {
    SourceName.PUBMED: 3,
    SourceName.CROSSREF: 2,
    SourceName.ARXIV: 1,
}

# Fields copied from a duplicate into the winner when the winner has none.
_FILLABLE_FIELDS = (
    "doi",
    "abstract",
    "published_date",
    "journal",
    "publisher",
    "volume",
    "issue",
    "pages",
    "publication_type",
    "pdf_url",
    "citation_count",
)


@dataclass
class _Cluster:
    """A set of records believed to describe the same work."""

    papers: list[Paper] = field(default_factory=list)

    def add(self, paper: Paper) -> None:
        self.papers.append(paper)


@dataclass
class DedupeResult:
    """Merged papers plus a count of how many records were collapsed."""

    papers: list[Paper]
    duplicates_merged: int


def deduplicate(papers: list[Paper]) -> DedupeResult:
    """Collapse records describing the same work into one merged ``Paper``.

    Input order is preserved for the surviving records, so a caller that
    fed in a meaningfully ordered list gets a meaningfully ordered list back.
    """
    clusters: list[_Cluster] = []
    by_doi: dict[str, int] = {}
    by_title: dict[str, int] = {}
    by_block: dict[str, list[int]] = {}

    for paper in papers:
        index = _find_cluster(paper, by_doi, by_title, by_block, clusters)
        if index is None:
            index = len(clusters)
            clusters.append(_Cluster())
        clusters[index].add(paper)
        _index_cluster(paper, index, by_doi, by_title, by_block)

    merged = [_merge_cluster(cluster) for cluster in clusters]
    duplicates = len(papers) - len(merged)
    if duplicates:
        logger.info("deduplicated %d records into %d papers", len(papers), len(merged))
    return DedupeResult(papers=merged, duplicates_merged=duplicates)


def _find_cluster(
    paper: Paper,
    by_doi: dict[str, int],
    by_title: dict[str, int],
    by_block: dict[str, list[int]],
    clusters: list[_Cluster],
) -> int | None:
    """Locate the existing cluster this paper belongs to, if any."""
    if paper.doi and paper.doi in by_doi:
        return by_doi[paper.doi]

    key = paper.normalized_title
    if not key:
        return None

    exact = by_title.get(key)
    if exact is not None and _doi_compatible(paper, clusters[exact]):
        return exact

    for candidate_index in by_block.get(key[:_BLOCK_PREFIX_LENGTH], []):
        if not _doi_compatible(paper, clusters[candidate_index]):
            continue
        for existing in clusters[candidate_index].papers:
            if _titles_match(key, existing.normalized_title):
                return candidate_index
    return None


def _doi_compatible(paper: Paper, cluster: _Cluster) -> bool:
    """Reject a title-based match when the DOIs positively disagree.

    Errata, corrections and re-publications routinely share a title with
    the work they refer to. Two records that both carry a DOI and disagree
    are, by definition, two different registered works.
    """
    if not paper.doi:
        return True
    return all(existing.doi in (None, paper.doi) for existing in cluster.papers)


def _index_cluster(
    paper: Paper,
    index: int,
    by_doi: dict[str, int],
    by_title: dict[str, int],
    by_block: dict[str, list[int]],
) -> None:
    """Record this paper's keys so later records can find its cluster."""
    if paper.doi:
        by_doi.setdefault(paper.doi, index)
    key = paper.normalized_title
    if key:
        by_title.setdefault(key, index)
        block = by_block.setdefault(key[:_BLOCK_PREFIX_LENGTH], [])
        if index not in block:
            block.append(index)


def _titles_match(left: str, right: str) -> bool:
    """Fuzzy title comparison, guarded by a cheap length check first."""
    if not left or not right:
        return False
    shorter, longer = sorted((len(left), len(right)))
    if shorter / longer < 0.8:
        return False
    return SequenceMatcher(None, left, right).ratio() >= _TITLE_SIMILARITY_THRESHOLD


def _merge_cluster(cluster: _Cluster) -> Paper:
    """Fold a cluster into one record, keeping the best value for each field."""
    if len(cluster.papers) == 1:
        return cluster.papers[0]

    ordered = sorted(
        cluster.papers,
        key=lambda p: (p.completeness(), _SOURCE_PRIORITY.get(p.source, 0)),
        reverse=True,
    )
    winner, *rest = ordered
    merged = winner.model_copy(deep=True)

    for other in rest:
        for name in _FILLABLE_FIELDS:
            if getattr(merged, name) is None:
                value = getattr(other, name)
                if value is not None:
                    setattr(merged, name, value)
        if not merged.authors and other.authors:
            merged.authors = list(other.authors)
        if not merged.keywords and other.keywords:
            merged.keywords = list(other.keywords)

    # Record every source that corroborated this work, in a stable order.
    corroborating: list[SourceName] = []
    for other in rest:
        if other.source != merged.source and other.source not in corroborating:
            corroborating.append(other.source)
    merged.also_found_in = corroborating

    # A DOI recovered from a sibling changes the record's canonical identity.
    if merged.doi and not merged.id.startswith("doi:"):
        merged.id = f"doi:{merged.doi}"
    return merged
