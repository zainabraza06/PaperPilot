"""Fill missing abstracts into the frozen evaluation pool.

The pool is deliberately frozen so ranking numbers are reproducible, and
re-running the fan-out to pick up abstracts would defeat that: the
literature moves, candidates change, and judgments for papers that no
longer appear become dead weight.

This fills abstracts into the *existing* records instead. Paper ids are
derived from the DOI, so nothing is re-identified and every judgment in
``judgments.json`` stays valid — which makes a before/after comparison on
exactly the same graded set possible.

Usage::

    python -m scripts.backfill_pool --dry-run     # what would change
    python -m scripts.backfill_pool               # write it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.core.logging import configure_logging
from app.models.paper import Paper
from app.services.enrichment.abstracts import AbstractBackfill

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
POOL = EVAL_DIR / "pool.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change and exit"
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    raw = json.loads(POOL.read_text(encoding="utf-8"))
    pools = {qid: [Paper.model_validate(item) for item in items] for qid, items in raw.items()}

    flat = [paper for papers in pools.values() for paper in papers]
    before = sum(1 for paper in flat if not paper.abstract)
    with_doi = sum(1 for paper in flat if not paper.abstract and paper.doi)
    print(f"pool            : {len(flat)} records across {len(pools)} queries")
    print(f"no abstract     : {before} ({before / len(flat):.0%})")
    print(f"  ...with a DOI : {with_doi}  (the only ones that can be looked up)")

    backfill = AbstractBackfill(get_settings())
    try:
        # Deduplicated across queries: the same paper appears in several
        # pools, and looking it up once per appearance would be rude to
        # an API that asks for politeness.
        unique: dict[str, Paper] = {}
        for paper in flat:
            if paper.doi and not paper.abstract:
                unique.setdefault(paper.doi, paper)
        filled, report = await backfill.fill(list(unique.values()))
    finally:
        await backfill.aclose()

    recovered = {paper.doi: paper.abstract for paper in filled if paper.abstract}
    print(f"\nrecovered       : {len(recovered)} of {with_doi} ({len(recovered) / max(with_doi, 1):.0%})")
    print(f"elapsed         : {report.elapsed_ms}ms")
    if report.reason:
        print(f"note            : {report.reason}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    changed = 0
    for papers in pools.values():
        for index, paper in enumerate(papers):
            if not paper.abstract and paper.doi and paper.doi in recovered:
                papers[index] = paper.model_copy(update={"abstract": recovered[paper.doi]})
                changed += 1

    after = sum(1 for papers in pools.values() for p in papers if not p.abstract)
    POOL.write_text(
        json.dumps(
            {qid: [p.model_dump(mode="json") for p in papers] for qid, papers in pools.items()},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nupdated {changed} pooled records ({before} -> {after} without an abstract)")
    print(f"wrote {POOL}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging("WARNING")
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
