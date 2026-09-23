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
from app.services.ranking.embeddings import build_embedder
from app.services.summarization.extractive import ExtractiveSummarizer
from app.services.summarization.grounding import GroundingChecker
from app.services.summarization.provider import build_provider
from app.services.summarization.quality import QualityScorer, QualityScores
from app.services.summarization.summarizer import PaperSummarizer
from app.services.summarization.support import SemanticSupportChecker, split_sentences

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
    parser.add_argument(
        "--min-support",
        type=float,
        default=None,
        help="Override the semantic support threshold (0 disables the check)",
    )
    parser.add_argument(
        "--sweep-support",
        action="store_true",
        help="Generate once, then score every support threshold over the same text",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Also score summary quality against the lead-3 and extractive baselines",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    model = args.model or settings.summary_model
    provider = build_provider(settings.llm_provider, settings.mistral_api_key, model)
    if provider is None:
        print("no provider configured; set PAPERPILOT_MISTRAL_API_KEY to measure generation")
        return 1

    papers = load_papers(args.limit)

    # The support check is what makes this measurement meaningful now: it
    # is the rule most likely to reject a legitimate model summary, so its
    # real cost only shows up against live generation.
    min_support = (
        args.min_support if args.min_support is not None else settings.grounding_min_support
    )
    support = (
        SemanticSupportChecker(build_embedder(settings.embedding_model), min_support=min_support)
        if min_support > 0
        else None
    )
    checker = GroundingChecker(
        min_overlap=settings.grounding_min_overlap, support=support
    )

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
    print(f"threshold : overlap >= {settings.grounding_min_overlap}, support >= {min_support}")
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

    # Advisories never block, so they are invisible in the outcome table
    # above — but they are the entire output of the support check on real
    # generated text, and the rate is what says whether it is usable.
    cautioned = [
        paper for paper in summarized if paper.summary and paper.summary.grounding.advisories
    ]
    if cautioned:
        print(
            f"\n{len(cautioned)} of {total} accepted summaries carry a support "
            f"caution ({len(cautioned) / total:.1%})"
        )

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

    if args.quality:
        _report_quality(summarized, settings.embedding_model)

    if args.sweep_support:
        _sweep_live(summarized, settings.embedding_model)

    print(f"\nreport: {report.model_dump_json(indent=2)}")
    close = getattr(provider, "aclose", None)
    if close is not None:
        await close()
    return 0


def _report_quality(papers: list[Paper], embedding_model: str) -> None:
    """Score the generated summaries, and two baselines, on the same abstracts.

    The baselines are the point. Every metric here is intrinsic, so its
    absolute value is close to meaningless — ``coverage=0.62`` is neither
    good nor bad until something else has been scored on the same papers.

    * **lead-3** — the first three sentences, verbatim. The standard
      summarization baseline, and a stubborn one, because abstracts are
      already written to put the important thing early.
    * **extractive** — the fallback this system ships, which picks
      sentences by title overlap and finding markers rather than by
      position.

    A generated summary earns its API call by beating both on coverage and
    lead bias at a lower compression ratio. If it does not, the honest
    conclusion is that the fallback was good enough, and this prints the
    numbers either way.
    """
    from app.services.ranking.cache import CachedEmbedder

    pairs = [
        (paper, paper.summary.text)
        for paper in papers
        if paper.summary and paper.abstract and paper.summary.is_ai_generated
    ]
    if not pairs:
        print("\nno generated summaries to score for quality")
        return

    embedder = CachedEmbedder(build_embedder(embedding_model), capacity=200_000)
    scorer = QualityScorer(embedder)
    extractive = ExtractiveSummarizer()

    systems: dict[str, list[QualityScores]] = {"generated": [], "lead-3": [], "extractive": []}
    for paper, generated in pairs:
        abstract = paper.abstract or ""
        candidates = {
            "generated": generated,
            "lead-3": " ".join(split_sentences(abstract)[:3]),
            "extractive": extractive.summarize(paper.title, abstract),
        }
        scored = {name: scorer.score(text, abstract) for name, text in candidates.items()}
        # Only keep papers every system could be scored on, so the columns
        # are averages over the same abstracts and can be compared.
        if any(value is None for value in scored.values()):
            continue
        for name, value in scored.items():
            assert value is not None
            systems[name].append(value)

    counted = len(systems["generated"])
    if not counted:
        print("\nno summaries could be scored for quality")
        return

    print(f"\nsummary quality on {counted} papers (all systems on the same abstracts):")
    names = list(systems)
    print(f"{'metric':<22}" + "".join(f"{name:>13}" for name in names))
    print("-" * (22 + 13 * len(names)))
    for field in QualityScores.fields():
        cells = "".join(
            f"{sum(getattr(s, field) for s in systems[name]) / counted:>13.3f}"
            for name in names
        )
        print(f"{field:<22}{cells}")
    print()
    print("  coverage/lead_bias: higher is better. compression: lower is better at")
    print("  equal coverage. novelty separates rewriting from copying — the")
    print("  extractive column is ~0 by construction and is the reference for it.")


def _sweep_live(papers: list[Paper], embedding_model: str) -> None:
    """Score every support threshold over summaries that were generated once.

    The synthetic sweep in ``evaluate_grounding`` measures false cautions
    against hand-built paraphrases, which turn out to be far easier than
    what a model actually writes. This measures the same thing against real
    generated text, which is the number that decides whether a caution
    means anything: a warning that fires on half of all summaries is noise
    a reader learns to ignore.

    Generation happens once and every threshold is scored over the same
    text, so the curve is not confounded by sampling differences.
    """
    from app.services.ranking.cache import CachedEmbedder

    pairs = [
        (paper.summary.text, paper.abstract)
        for paper in papers
        if paper.summary and paper.abstract and paper.summary.is_ai_generated
    ]
    if not pairs:
        print("\nno generated summaries to sweep")
        return

    embedder = CachedEmbedder(build_embedder(embedding_model), capacity=200_000)
    print(f"\nsupport threshold vs caution rate on {len(pairs)} real summaries:")
    print(f"{'support':>8} {'cautioned':>11} {'rate':>8}")
    print("-" * 30)
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        support = SemanticSupportChecker(embedder, min_support=threshold)
        flagged = sum(1 for text, abstract in pairs if support.unsupported(text, abstract))
        print(f"{threshold:>8.2f} {flagged:>11} {flagged / len(pairs):>7.1%}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging("WARNING")
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
