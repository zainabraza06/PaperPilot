"""Live demo of the retrieval and ranking pipeline.

Runs one real query against PubMed, arXiv and Crossref concurrently, then
ranks the merged set, and prints everything the pipeline decided along the
way: how the query was classified, what each source did, how many
duplicates collapsed, and the relevance score behind each position.

Usage::

    python -m scripts.demo_search "prime editing in primary human cells"
    python -m scripts.demo_search "10.1038/s41587-022-01234-5"
    python -m scripts.demo_search "CRISPR" --sources arxiv crossref
    python -m scripts.demo_search "diffusion models" --strategy semantic
    python -m scripts.demo_search "diffusion models" --no-rank   # A/B the ranker
    python -m scripts.demo_search "diffusion models" --no-enrich # skip NER/clusters
    python -m scripts.demo_search "diffusion models" --no-summary
    python -m scripts.demo_search "prime editing" --export bibtex --show 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap

from app.config import get_settings
from app.core.logging import configure_logging
from app.models.paper import Paper, SourceName
from app.models.search import SearchRequest, SearchResponse, SourceStatus
from app.services.enrichment.clustering import TopicClusterer
from app.services.enrichment.entities import build_entity_extractor
from app.services.export import ExportFormat, get_formatter
from app.services.ranking.cache import CachedEmbedder
from app.services.ranking.embeddings import build_embedder
from app.services.ranking.hybrid import FusionStrategy, HybridRanker
from app.services.search_service import SearchService
from app.services.summarization.grounding import GroundingChecker
from app.services.summarization.provider import build_provider
from app.services.summarization.summarizer import PaperSummarizer
from app.sources.registry import SourceRegistry
from app.storage.summary_cache import SummaryCache

# Status glyphs, chosen so the output stays readable in a plain terminal.
_STATUS_MARK = {
    SourceStatus.OK: "OK  ",
    SourceStatus.EMPTY: "NONE",
    SourceStatus.TIMEOUT: "SLOW",
    SourceStatus.RATE_LIMITED: "RATE",
    SourceStatus.UNAVAILABLE: "DOWN",
    SourceStatus.PARSE_ERROR: "PARS",
    SourceStatus.SKIPPED: "SKIP",
}

_SOURCE_BADGE = {
    SourceName.PUBMED: "PubMed",
    SourceName.ARXIV: "arXiv",
    SourceName.CROSSREF: "Crossref",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="Topic, keyword, DOI, arXiv id, PMID, or a pasted abstract")
    parser.add_argument("--limit", type=int, default=10, help="Results per source (default: 10)")
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=[s.value for s in SourceName],
        help="Restrict the fan-out (default: all three)",
    )
    parser.add_argument("--show", type=int, default=10, help="How many merged papers to print")
    parser.add_argument("--verbose", action="store_true", help="Show connector log output")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in FusionStrategy],
        help="Fusion strategy (default: from settings)",
    )
    parser.add_argument("--no-rank", action="store_true", help="Skip ranking entirely")
    parser.add_argument("--no-enrich", action="store_true", help="Skip NER and clustering")
    parser.add_argument("--entities", type=int, default=4, help="Entities to show per paper")
    parser.add_argument("--no-summary", action="store_true", help="Skip summarization")
    parser.add_argument("--no-cache", action="store_true", help="Ignore the summary cache")
    parser.add_argument(
        "--export",
        choices=[f.value for f in ExportFormat],
        help="Also print citations for the shown papers in this format",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> SearchResponse:
    settings = get_settings()
    registry = SourceRegistry(settings)

    # One embedder, shared: the ranker and the clusterer see the same papers,
    # so the second one to run reads its vectors out of the cache.
    embedder = CachedEmbedder(build_embedder(settings.embedding_model))

    ranker = None
    if not args.no_rank:
        strategy = FusionStrategy(args.strategy or settings.ranking_strategy)
        ranker = HybridRanker(embedder, strategy=strategy, alpha=settings.ranking_alpha)

    extractor = None if args.no_enrich else build_entity_extractor(settings.ner_model)
    clusterer = (
        None
        if args.no_enrich
        else TopicClusterer(
            embedder,
            max_clusters=settings.max_clusters,
            min_papers=settings.min_papers_to_cluster,
        )
    )
    summarizer = None
    if not args.no_summary:
        cache = None if args.no_cache else SummaryCache(settings.summary_cache_path)
        summarizer = PaperSummarizer(
            build_provider(
                settings.llm_provider, settings.mistral_api_key, settings.summary_model
            ),
            checker=GroundingChecker(min_overlap=settings.grounding_min_overlap),
            cache=cache,
            max_concurrent=settings.summary_max_concurrent,
            max_attempts=settings.summary_max_attempts,
        )

    service = SearchService(
        registry, settings, ranker, extractor, clusterer, summarizer
    )
    try:
        return await service.search(
            SearchRequest(
                query=args.query,
                limit_per_source=args.limit,
                sources=[SourceName(s) for s in args.sources] if args.sources else None,
            )
        )
    finally:
        await registry.aclose()


def print_response(response: SearchResponse, show: int, entity_limit: int = 4) -> None:
    print()
    print("=" * 78)
    print(f"QUERY      {response.query.raw[:200]}")
    print(f"INTENT     {response.query.intent.value}", end="")
    if response.query.identifier_kind:
        print(f" ({response.query.identifier_kind.value}: {response.query.identifier})", end="")
    print()
    if response.query.search_terms != response.query.raw:
        print(f"SENT AS    {response.query.search_terms}")
    print("=" * 78)

    print("\nSOURCES")
    for report in response.sources:
        mark = _STATUS_MARK[report.status]
        line = f"  [{mark}] {_SOURCE_BADGE[report.source]:<9} {report.returned:>3} results  {report.elapsed_ms:>5} ms"
        if report.message:
            line += f"  — {report.message}"
        print(line)

    print(
        f"\nMERGED     {response.total} papers "
        f"({response.duplicates_merged} duplicate record(s) collapsed) "
        f"in {response.elapsed_ms} ms"
    )
    if response.degraded:
        print("           partial results: at least one source failed (see above)")

    ranking = response.ranking
    if ranking.applied:
        print(
            f"RANKED     {ranking.strategy} fusion on {ranking.model} "
            f"in {ranking.elapsed_ms} ms"
        )
    else:
        print(f"UNRANKED   retrieval order ({ranking.reason})")

    entities = response.entities
    if entities.applied:
        print(f"ENTITIES   {entities.entities_found} found via {entities.model} "
              f"in {entities.elapsed_ms} ms")
    else:
        print(f"ENTITIES   none ({entities.reason})")

    summaries = response.summaries
    if summaries.applied:
        line = (
            f"SUMMARIES  {summaries.summarized} via {summaries.model} "
            f"in {summaries.elapsed_ms} ms"
        )
        details = []
        if summaries.from_cache:
            details.append(f"{summaries.from_cache} cached")
        if summaries.regenerated:
            details.append(f"{summaries.regenerated} regenerated after failing grounding")
        if summaries.fell_back:
            details.append(f"{summaries.fell_back} fell back to extractive")
        if details:
            line += "  (" + ", ".join(details) + ")"
        print(line)
        if summaries.reason:
            print(f"           {summaries.reason}")
    else:
        print(f"SUMMARIES  none ({summaries.reason})")

    clustering = response.clustering
    if clustering.applied:
        print(f"CLUSTERS   {clustering.clusters} sub-topics "
              f"(silhouette {clustering.silhouette}, {clustering.elapsed_ms} ms)")
        for cluster in response.clusters:
            print(f"             [{cluster.size:>2}] {cluster.label}")
    else:
        print(f"CLUSTERS   none ({clustering.reason})")

    if not response.papers:
        print("\nNo papers found.")
        return

    print("\n" + "-" * 78)
    for index, paper in enumerate(response.papers[:show], start=1):
        print_paper(index, paper, entity_limit)


def print_paper(index: int, paper: Paper, entity_limit: int = 4) -> None:
    badges = "+".join(_SOURCE_BADGE[s] for s in paper.all_sources)
    year = paper.published_date.isoformat() if paper.published_date else "no date"

    print(f"\n{index:>2}. {textwrap.shorten(paper.title, width=72, placeholder=' …')}")
    if paper.score:
        # A bar rather than a bare number, with the two components beside it:
        # this is how the frontend will present relevance too.
        filled = round(paper.score.combined * 20)
        bar = "#" * filled + "." * (20 - filled)
        print(
            f"    [{bar}] {paper.score.combined:.2f}"
            f"   semantic={paper.score.semantic:+.3f}  bm25={paper.score.lexical:.2f}"
        )
    print(f"    {badges}  |  {year}  |  {paper.journal or 'no venue'}")
    print(f"    {_format_authors(paper)}")
    print(f"    DOI: {paper.doi or '—'}")
    if paper.summary:
        mark = {
            "generated": "AI",
            "regenerated": "AI*",
            "extractive": "EXT",
        }[paper.summary.origin.value]
        status = paper.summary.grounding.status.value
        print(f"    [{mark}|{status}] {textwrap.shorten(paper.summary.text, width=66, placeholder=' …')}")
        for issue in paper.summary.grounding.issues[:2]:
            print(f"        ! {issue.detail}")
    elif paper.abstract:
        print(f"    {textwrap.shorten(paper.abstract, width=72, placeholder=' …')}")
    else:
        print("    (no abstract deposited)")
    if paper.entities:
        shown = ", ".join(
            f"{entity.text} ({entity.label.value})" for entity in paper.entities[:entity_limit]
        )
        print(f"    entities: {shown}")


def print_citations(response: SearchResponse, show: int, export_format: ExportFormat) -> None:
    """Print the shown papers as a citation file.

    Exactly what the export endpoint would return, so the formats can be
    eyeballed - and pasted into a reference manager - without running the
    API.
    """
    formatter = get_formatter(export_format)
    print("\n" + "=" * 78)
    print(f"CITATIONS  {export_format.value}  (.{formatter.extension})")
    print("=" * 78)
    # RIS uses CRLF; a terminal shows the stray carriage returns otherwise.
    print(formatter.format_many(response.papers[:show]).replace("\r\n", "\n"))


def _format_authors(paper: Paper, limit: int = 3) -> str:
    if not paper.authors:
        return "no authors listed"
    names = [author.name for author in paper.authors[:limit]]
    suffix = f" +{len(paper.authors) - limit} more" if len(paper.authors) > limit else ""
    return ", ".join(names) + suffix


def main(argv: list[str] | None = None) -> int:
    # Author names and abstracts are full Unicode; the Windows console
    # defaults to a legacy codepage and would mangle them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    configure_logging("INFO" if args.verbose else "WARNING")
    response = asyncio.run(run(args))
    print_response(response, args.show, args.entities)
    if args.export:
        print_citations(response, args.show, ExportFormat(args.export))
    # A search that reached no source at all is a failure worth an exit code.
    return 0 if any(not r.status.is_failure for r in response.sources) else 1


if __name__ == "__main__":
    sys.exit(main())
