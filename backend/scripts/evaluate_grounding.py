"""Measure the grounding checker on real abstracts.

A fact-checking layer that is never itself checked is decoration. This
harness turns the frozen golden-set abstracts into a labelled dataset and
reports, per failure mode, how much of it the checker actually catches.

**How the labels are made.** Positives are extractive summaries — sentences
lifted verbatim from the abstract — which are grounded by construction, so
any rejection is a false positive. Negatives are those same summaries with
one specific, deterministic corruption applied: a fabricated statistic, an
invented method name, a reversed direction of effect, an unsupported
novelty claim, or a summary of an entirely different paper. Each
corruption maps to exactly one issue kind, so a miss is attributable.

**Why the headline numbers are not the interesting ones.** Each corruption
is a clean instance of exactly the failure mode one rule was written to
catch, so a high recall on them mostly confirms the rules fire, not that
real hallucinations get caught. Two buckets exist to make the measurement
say something:

* **paraphrase** positives — grounded summaries reworded away from the
  abstract's exact sentences. This is where false positives actually
  appear, because a real generated summary shares less vocabulary with its
  source than a verbatim extract does.
* **recombination** negatives — unsupported claims built entirely from the
  abstract's own words, with no fabricated number, entity or reversed
  direction. These are *expected to pass*. They quantify the documented
  blind spot: the check is lexical, not inferential. The number to read is
  how many slip through, and it should be most of them.

So: recall on the designed corruptions is an upper bound, the paraphrase
false-positive rate is the honest cost of the check, and the recombination
row is the honest limit of it.

Usage::

    python -m scripts.evaluate_grounding
    python -m scripts.evaluate_grounding --show-misses
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from app.config import get_settings
from app.models.paper import Paper
from app.models.summary import IssueKind
from app.services.ranking.cache import CachedEmbedder
from app.services.ranking.embeddings import build_embedder
from app.services.summarization.attacks import build_attacks
from app.services.summarization.extractive import ExtractiveSummarizer
from app.services.summarization.grounding import GroundingChecker
from app.services.summarization.support import SemanticSupportChecker

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
SEED = 20240905

_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

#: Method names that appear in no abstract in the pool.
_FAKE_ENTITIES = ("CRISPRoff", "OmniFold-7", "TransBERT-XL", "scVelo-Pro", "QuantumNet")

_DIRECTION_FLIPS = {
    "increased": "decreased",
    "decreased": "increased",
    "higher": "lower",
    "lower": "higher",
    "improved": "worsened",
    "reduced": "elevated",
    "more": "less",
}

_OVERCLAIMS = (
    "This is the first study to prove the mechanism. ",
    "The method is definitively superior to all alternatives. ",
)


class Case(NamedTuple):
    """One labelled example."""

    paper_id: str
    summary: str
    abstract: str
    #: ``None`` for a summary that should be accepted, otherwise the issue
    #: that should fire.
    expected: IssueKind | None
    label: str
    #: True when the checker is *expected* to miss this - the blind spot.
    out_of_scope: bool = False


#: Grounding-preserving rewrites, applied to build the paraphrase positives.
_REWRITES = (
    ("We ", "The authors "),
    ("we ", "the authors "),
    ("Here ", ""),
    ("In this study, ", ""),
    ("These results ", "The findings "),
    ("demonstrate", "show"),
    ("utilized", "used"),
    ("Our ", "Their "),
)

_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


def load_papers() -> list[Paper]:
    raw = json.loads((EVAL_DIR / "pool.json").read_text(encoding="utf-8"))
    papers = [Paper.model_validate(item) for items in raw.values() for item in items]
    # Only papers with a substantial abstract can be corrupted meaningfully.
    return [p for p in papers if p.abstract and len(p.abstract) > 300]


def build_cases(papers: list[Paper]) -> list[Case]:
    """Turn each abstract into one positive and up to five negatives."""
    rng = random.Random(SEED)
    extractive = ExtractiveSummarizer()
    cases: list[Case] = []

    for index, paper in enumerate(papers):
        abstract = paper.abstract or ""
        faithful = extractive.summarize(paper.title, abstract)
        cases.append(Case(paper.id, faithful, abstract, None, "faithful"))

        # 1. A statistic that is nowhere in the abstract.
        numbers = _NUMBER.findall(faithful)
        if numbers:
            original = numbers[0]
            fake = str(int(float(original)) + rng.randint(11, 89))
            cases.append(
                Case(
                    paper.id,
                    faithful.replace(original, fake, 1),
                    abstract,
                    IssueKind.FABRICATED_NUMBER,
                    "fabricated number",
                )
            )

        # 2. A method the paper never used.
        entity = _FAKE_ENTITIES[index % len(_FAKE_ENTITIES)]
        if entity.lower() not in abstract.lower():
            cases.append(
                Case(
                    paper.id,
                    f"The authors applied {entity}. {faithful}",
                    abstract,
                    IssueKind.FABRICATED_ENTITY,
                    "fabricated entity",
                )
            )

        # 3. A reversed direction of effect.
        flipped = _flip_direction(faithful, abstract)
        if flipped:
            cases.append(
                Case(
                    paper.id,
                    flipped,
                    abstract,
                    IssueKind.CONTRADICTED_DIRECTION,
                    "reversed direction",
                )
            )

        # 4. A novelty or certainty claim the abstract does not make.
        overclaim = _OVERCLAIMS[index % len(_OVERCLAIMS)]
        if not any(word in abstract.lower() for word in ("first", "prove", "superior")):
            cases.append(
                Case(
                    paper.id,
                    overclaim + faithful,
                    abstract,
                    IssueKind.OVERCLAIM,
                    "overclaim",
                )
            )

        # 5. A summary of a completely different paper.
        other = papers[(index + len(papers) // 2) % len(papers)]
        if other.id != paper.id and other.abstract:
            cases.append(
                Case(
                    paper.id,
                    extractive.summarize(other.title, other.abstract),
                    abstract,
                    IssueKind.LOW_OVERLAP,
                    "wrong paper",
                )
            )

        # 6. A grounded summary reworded away from the abstract's wording.
        #    Should still be accepted; this is where false positives live.
        paraphrase = _paraphrase(faithful)
        if paraphrase != faithful:
            cases.append(Case(paper.id, paraphrase, abstract, None, "paraphrase"))

        # 7. Claims assembled from the abstract's own sentences. Every
        #    lexical rule passes these by construction, so they isolate
        #    exactly what the semantic support check is for.
        for attack in build_attacks(abstract, rng):
            cases.append(
                Case(
                    paper.id,
                    attack.text,
                    abstract,
                    IssueKind.UNSUPPORTED_CLAIM,
                    f"recombination: {attack.family}",
                )
            )

    return cases


def _paraphrase(summary: str) -> str:
    """Reword without adding anything the abstract does not support."""
    text = _PARENTHETICAL.sub("", summary)
    for old, new in _REWRITES:
        text = text.replace(old, new)
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) > 1:
        # Reordering changes nothing about what is claimed.
        sentences = sentences[1:] + sentences[:1]
    return ". ".join(sentences).rstrip(".") + "."




def _flip_direction(summary: str, abstract: str) -> str | None:
    """Reverse a directional word, but only one the abstract really states."""
    lowered_abstract = abstract.lower()
    for word, opposite in _DIRECTION_FLIPS.items():
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        if pattern.search(summary) and opposite not in lowered_abstract:
            return pattern.sub(opposite, summary, count=1)
    return None


def evaluate(cases: list[Case], checker: GroundingChecker) -> dict[str, dict[str, int]]:
    """Score the checker, bucketed by case type.

    ``caught`` means "the checker did the right thing": accepted, for a
    case that should be accepted; flagged with the expected issue, for one
    that should not.
    """
    tallies: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "caught": 0})
    for case in cases:
        verdict = checker.check(case.summary, case.abstract)
        bucket = tallies[case.label]
        bucket["total"] += 1
        if case.expected is None:
            bucket["caught"] += int(verdict.passed)
        else:
            kinds = {issue.kind for issue in verdict.issues}
            bucket["caught"] += int(case.expected in kinds)
    return dict(tallies)


def _print_row(label: str, tallies: dict[str, dict[str, int]]) -> None:
    bucket = tallies.get(label)
    if not bucket or not bucket["total"]:
        return
    rate = bucket["caught"] / bucket["total"]
    print(f"  {label:<24} {bucket['total']:>4} {bucket['caught']:>9} {rate:>6.1%}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-misses", action="store_true", help="Print undetected cases")
    parser.add_argument("--min-overlap", type=float, default=None, help="Override the threshold")
    parser.add_argument(
        "--no-support",
        action="store_true",
        help="Disable the semantic support check, to measure the lexical rules alone",
    )
    parser.add_argument(
        "--sweep-support",
        action="store_true",
        help="Score several support thresholds and print the trade-off",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    settings = get_settings()
    papers = load_papers()
    cases = build_cases(papers)

    support = None
    if not args.no_support:
        # Reuses the embedder the app already loads; no second model.
        support = SemanticSupportChecker(
            CachedEmbedder(build_embedder(settings.embedding_model), capacity=200_000)
        )

    overlap = args.min_overlap if args.min_overlap is not None else settings.grounding_min_overlap

    if args.sweep_support:
        _sweep(cases, overlap)
        return 0

    checker = GroundingChecker(min_overlap=overlap, support=support)
    tallies = evaluate(cases, checker)

    print(f"abstracts : {len(papers)}")
    print(f"cases     : {len(cases)}")
    print()
    print(f"  {'case type':<24} {'n':>4} {'correct':>9} {'rate':>7}")
    print("-" * 52)

    positives = ("faithful", "paraphrase")
    designed = ("fabricated number", "fabricated entity", "reversed direction",
                "overclaim", "wrong paper")
    recombination = tuple(sorted({c.label for c in cases if c.label.startswith("recombination")}))

    print("should be ACCEPTED:")
    for label in positives:
        _print_row(label, tallies)
    print("\nshould be REJECTED (one rule each):")
    for label in designed:
        _print_row(label, tallies)
    print("\nrecombination - built from the abstract's own sentences, so every")
    print("lexical rule passes them. Only the support check can see these:")
    for label in recombination:
        _print_row(label, tallies)

    caught = sum(tallies.get(label, {}).get("caught", 0) for label in designed)
    total = sum(tallies.get(label, {}).get("total", 0) for label in designed)
    accepted = sum(tallies.get(label, {}).get("caught", 0) for label in positives)
    positive_total = sum(tallies.get(label, {}).get("total", 0) for label in positives)
    blind = {
        "total": sum(tallies.get(label, {}).get("total", 0) for label in recombination),
        "caught": sum(tallies.get(label, {}).get("caught", 0) for label in recombination),
    }

    print("\n" + "-" * 52)
    print(f"{'recall on designed cases':<26} {caught:>4}/{total:<5} {caught / max(total, 1):>6.1%}")
    print(
        f"{'false-positive rate':<26} "
        f"{positive_total - accepted:>4}/{positive_total:<5} "
        f"{(positive_total - accepted) / max(positive_total, 1):>6.1%}"
    )
    print(
        f"{'recombination caught':<26} "
        f"{blind['caught']:>4}/{blind['total']:<5} "
        f"{blind['caught'] / max(blind['total'], 1):>6.1%}"
    )

    if args.show_misses:
        print("\nundetected corruptions:")
        for case in cases:
            if case.expected is None or case.out_of_scope:
                continue
            kinds = {i.kind for i in checker.check(case.summary, case.abstract).issues}
            if case.expected not in kinds:
                print(f"  [{case.label}] {case.summary[:110]}")
    return 0


def _sweep(cases: list[Case], overlap: float) -> None:
    """Score several support thresholds, so the default is a measured choice.

    The trade-off is the whole story: raising the threshold catches more
    recombinations and rejects more honest paraphrase. Printing both
    columns is what makes the chosen value defensible rather than tuned to
    whichever number looked best.
    """
    from app.config import get_settings

    embedder = CachedEmbedder(build_embedder(get_settings().embedding_model), capacity=200_000)
    recombination = tuple(sorted({c.label for c in cases if c.label.startswith("recombination")}))

    # Acceptance no longer moves with this threshold - support issues are
    # advisory, so they never block. What moves is how often a legitimate
    # summary gets an unnecessary caution, which is the real cost now.
    print(f"{'support':>8} {'recombination flagged':>22} {'false cautions':>16}")
    print("-" * 50)
    for threshold in (0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75):
        checker = GroundingChecker(
            min_overlap=overlap,
            support=SemanticSupportChecker(embedder, min_support=threshold),
        )
        tallies = evaluate(cases, checker)

        caught = sum(tallies.get(name, {}).get("caught", 0) for name in recombination)
        total = sum(tallies.get(name, {}).get("total", 0) for name in recombination)

        # A caution on a summary that is actually faithful is the false
        # positive that matters, so it is counted directly.
        legitimate = [c for c in cases if c.expected is None]
        cautioned = sum(
            1
            for case in legitimate
            if any(
                issue.kind is IssueKind.UNSUPPORTED_CLAIM
                for issue in checker.check(case.summary, case.abstract).issues
            )
        )
        print(
            f"{threshold:>8.2f} {caught / max(total, 1):>21.1%} "
            f"{cautioned / max(len(legitimate), 1):>15.1%}"
        )


if __name__ == "__main__":
    sys.exit(main())
