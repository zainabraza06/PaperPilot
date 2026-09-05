"""Measure live summary generation against the grounding check.

The gap the other harnesses leave: ``evaluate_grounding`` scores the
checker on synthetic corruptions, and the unit tests script the provider.
Neither says how often a *real* model, on *real* abstracts, writes
something the abstract does not support. This does.

For each paper it runs the full policy — generate, check, correct on
failure, fall back if the correction also fails — and reports where the
summaries ended up and which rules rejected them. That last breakdown is
the useful part: it says whether a model's failures are fabricated
statistics, invented method names, or drift, which is actionable in a way
that a single pass rate is not.

Caching is bypassed, since a cache hit would measure nothing.

Usage::

    python -m scripts.evaluate_summaries --limit 20
    python -m scripts.evaluate_summaries --model ministral-3b-latest --limit 20
    python -m scripts.evaluate_summaries --limit 10 --show-failures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

from app.config import get_settings
from app.core.logging import configure_logging
from app.models.paper import Paper
from app.models.summary import SummaryOrigin
from app.services.summarization.grounding import GroundingChecker
from app.services.summarization.provider import build_provider
from app.services.summarization.summarizer import PaperSummarizer

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def load_papers(limit: int) -> list[Paper]:
    """Distinct papers with abstracts long enough to be worth summarizing."""
    raw = json.loads((EVAL_DIR / "pool.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    papers: list[Paper] = []
    # Round-robin across queries so the sample spans every domain in the
    # pool rather than 20 papers about one topic.
    pools = [[Paper.model_validate(item) for item in items] for items in raw.values()]
    for index in range(max(len(pool) for pool in pools)):
        for pool in pools:
            if index >= len(pool):
                continue
            paper = pool[index]
            if paper.id in seen or not paper.abstract or len(paper.abstract) < 400:
                continue
            seen.add(paper.id)
            papers.append(paper)
            if len(papers) >= limit:
                return papers
    return papers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override the configured summary model")
    parser.add_argument("--limit", type=int, default=20, help="Papers to summarize")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--show-failures", action="store_true", help="Print rejected text")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    model = args.model or settings.summary_model
    provider = build_provider(settings.llm_provider, settings.mistral_api_key, model)
    if provider is None:
        print("no provider configured; set PAPERPILOT_MISTRAL_API_KEY to measure generation")
        return 1

    papers = load_papers(args.limit)
    checker = GroundingChecker(min_overlap=settings.grounding_min_overlap)

    # No cache: a hit would measure nothing.
    summarizer = PaperSummarizer(
        provider,
        checker=checker,
        cache=None,
        max_concurrent=args.concurrency,
        max_attempts=settings.summary_max_attempts,
    )

    print(f"model     : {model}")
    print(f"papers    : {len(papers)}")
    print(f"threshold : overlap >= {settings.grounding_min_overlap}")
    print()

    started = time.perf_counter()
    summarized, report = await summarizer.summarize(papers)
    elapsed = time.perf_counter() - started

    origins = Counter(p.summary.origin for p in summarized if p.summary)
    first_pass = origins[SummaryOrigin.GENERATED]
    corrected = origins[SummaryOrigin.REGENERATED]
    fell_back = origins[SummaryOrigin.EXTRACTIVE]
    total = len(summarized)

    print(f"{'outcome':<38} {'n':>4} {'share':>7}")
    print("-" * 52)
    print(f"{'grounded on first attempt':<38} {first_pass:>4} {first_pass / total:>6.1%}")
    print(f"{'rescued by the correcting retry':<38} {corrected:>4} {corrected / total:>6.1%}")
    print(f"{'fell back to extractive':<38} {fell_back:>4} {fell_back / total:>6.1%}")
    print("-" * 52)
    accepted = first_pass + corrected
    print(f"{'generated text accepted':<38} {accepted:>4} {accepted / total:>6.1%}")
    print(f"\n{elapsed:.1f}s wall clock, {elapsed / max(total, 1):.2f}s per paper")

    # Which rules actually fired, across every rejected attempt. A single
    # pass rate says a model failed; this says how.
    rejections: Counter[str] = Counter()
    failures: list[tuple[str, str, list[str]]] = []
    for paper in summarized:
        # The *first attempt's* issues, not the final text's - the final
        # text passed by definition, so re-checking it says nothing.
        if paper.summary is None or not paper.summary.rejected_for:
            continue
        for issue in paper.summary.rejected_for:
            rejections[issue.kind.value] += 1
        failures.append(
            (
                paper.id,
                paper.summary.text,
                [issue.detail for issue in paper.summary.rejected_for],
            )
        )

    if rejections:
        rejected_papers = len(failures)
        print(
            f"\nwhy the first attempt was rejected "
            f"({rejected_papers} of {total} papers needed one):"
        )
        for kind, count in rejections.most_common():
            print(f"  {kind:<26} {count:>3}")

    if args.show_failures and failures:
        print("\nrejected summaries:")
        for paper_id, text, details in failures[:10]:
            print(f"\n  {paper_id}")
            print(f"    {text[:160]}")
            for detail in details[:3]:
                print(f"    ! {detail}")

    print(f"\nreport: {report.model_dump_json(indent=2)}")
    close = getattr(provider, "aclose", None)
    if close is not None:
        await close()
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging("WARNING")
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
