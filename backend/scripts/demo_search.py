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
from app.services.ranking.embeddings import build_embedder
from app.services.ranking.hybrid import FusionStrategy, HybridRanker
from app.services.search_service import SearchService
from app.sources.registry import SourceRegistry

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
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> SearchResponse:
    settings = get_settings()
    registry = SourceRegistry(settings)

    ranker = None
    if not args.no_rank:
        strategy = FusionStrategy(args.strategy or settings.ranking_strategy)
        ranker = HybridRanker(
            build_embedder(settings.embedding_model),
            strategy=strategy,
            alpha=settings.ranking_alpha,
        )
    service = SearchService(registry, settings, ranker)
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


def print_response(response: SearchResponse, show: int) -> None:
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

    if not response.papers:
        print("\nNo papers found.")
        return

    print("\n" + "-" * 78)
    for index, paper in enumerate(response.papers[:show], start=1):
        print_paper(index, paper)


def print_paper(index: int, paper: Paper) -> None:
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
    if paper.abstract:
        print(f"    {textwrap.shorten(paper.abstract, width=72, placeholder=' …')}")
    else:
        print("    (no abstract deposited)")


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
    print_response(response, args.show)
    # A search that reached no source at all is a failure worth an exit code.
    return 0 if any(not r.status.is_failure for r in response.sources) else 1


if __name__ == "__main__":
    sys.exit(main())
