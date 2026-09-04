"""Harvest the evaluation candidate pool from the live APIs.

Run once (or when the pool needs refreshing). It freezes a real,
reproducible candidate set to disk so that ranking can be evaluated
offline and deterministically afterwards — the numbers in the README come
from re-ranking *this* frozen pool, not from a fresh network call whose
results would drift between runs.

Methodology note, and the reason this script exists at all: the pool is
the complete retrieval output for each query, and **every** paper in it
gets judged. There is no top-k pooling, so no strategy can be favoured by
having contributed more of the judged documents. The consequence is that
the reported recall is *recall within the retrieved candidate set* — it
measures re-ranking quality, not the retrieval layer's coverage of the
literature. Those are different claims and the README says which one it
is making.

Usage::

    python -m scripts.build_golden_set
    python -m scripts.build_golden_set --per-source 8 --refresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.core.logging import configure_logging
from app.models.search import SearchRequest
from app.services.search_service import SearchService
from app.sources.registry import SourceRegistry

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUERIES_PATH = EVAL_DIR / "queries.json"
POOL_PATH = EVAL_DIR / "pool.json"
TEMPLATE_PATH = EVAL_DIR / "judgments.template.json"

# Fields kept in the frozen pool. Everything the ranker reads, plus enough
# bibliographic detail for a human to assign a grade without re-fetching.
_POOL_FIELDS = (
    "id",
    "doi",
    "source",
    "source_id",
    "title",
    "abstract",
    "published_date",
    "url",
    "journal",
    "keywords",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-source", type=int, default=8, help="Results per source per query")
    parser.add_argument("--refresh", action="store_true", help="Overwrite an existing pool")
    return parser.parse_args(argv)


async def build(per_source: int) -> dict[str, list[dict[str, object]]]:
    settings = get_settings()
    registry = SourceRegistry(settings)
    # No ranker: the pool is the *retrieval* output. Ranking is what we are
    # evaluating, so it must not influence which papers get judged.
    service = SearchService(registry, settings, ranker=None)

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    pool: dict[str, list[dict[str, object]]] = {}
    try:
        for entry in queries:
            response = await service.search(
                SearchRequest(query=entry["query"], limit_per_source=per_source)
            )
            pool[entry["id"]] = [
                paper.model_dump(mode="json", include=set(_POOL_FIELDS))
                for paper in response.papers
            ]
            statuses = ", ".join(f"{r.source.value}={r.returned}" for r in response.sources)
            print(f"{entry['id']:<28} {len(response.papers):>3} candidates  ({statuses})")
    finally:
        await registry.aclose()
    return pool


def write_template(pool: dict[str, list[dict[str, object]]]) -> None:
    """Emit an unjudged skeleton, so grading is filling in blanks."""
    template = {
        query_id: {str(paper["id"]): 0 for paper in papers} for query_id, papers in pool.items()
    }
    TEMPLATE_PATH.write_text(json.dumps(template, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("WARNING")

    if POOL_PATH.exists() and not args.refresh:
        print(f"{POOL_PATH} already exists; pass --refresh to rebuild it.")
        return 1

    pool = asyncio.run(build(args.per_source))
    POOL_PATH.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    write_template(pool)

    total = sum(len(papers) for papers in pool.values())
    print(f"\nwrote {total} candidates across {len(pool)} queries to {POOL_PATH.name}")
    print(f"judgment skeleton written to {TEMPLATE_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
