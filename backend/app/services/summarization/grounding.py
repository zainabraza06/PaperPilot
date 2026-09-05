"""Checking that a summary only says what its abstract supports.

This is the part of Stage 4 that matters. Generating three fluent
sentences about a paper is easy and a language model will do it whether or
not it read the abstract carefully; the value of the feature rests
entirely on whether a researcher can trust what they read. So every
generated summary is checked, and one that fails is regenerated or
replaced rather than shown.

**The check is deliberately not an LLM.** Asking a model to grade its own
output is circular, doubles the cost and latency, and produces a verdict
that cannot be unit-tested. These checks are deterministic, run in
microseconds, and each one is a property that can be asserted:

1. **Fabricated numbers** — every figure in the summary must appear in the
   abstract. Invented statistics are the highest-impact hallucination in a
   research tool and the easiest thing in the world to verify.
2. **Fabricated entities** — every gene symbol, acronym and named method
   in the summary must appear in the abstract. Reuses the Stage 3 pattern
   extractor, so both stages agree on what a technical term is.
3. **Vocabulary overlap** — a summary of *this* abstract shares most of
   its content words with it. Low overlap means the model wrote about
   something else.
4. **Direction of effect** — "increased" must not become "decreased".
   A narrow check against a fixed list of antonym pairs, but it catches
   the one paraphrase error that would actively mislead a reader.
5. **Overclaiming** — novelty and certainty language ("the first",
   "proves", "cures") that the abstract itself does not use.
6. **Format** — the 2-3 sentences the prompt asked for.

**What this does not catch**, stated plainly because a grounding check
that oversells itself is worse than none: it is lexical, not inferential.
A fluent summary that recombines the abstract's own words into a claim the
abstract never made will pass. Catching that needs entailment, which is a
model, which brings back every problem listed above. The checks here cover
the failure modes that are both common and verifiable, and the limitation
is documented rather than hidden.
"""

from __future__ import annotations

import re

from app.models.summary import (
    GroundingIssue,
    GroundingStatus,
    GroundingVerdict,
    IssueKind,
)
from app.services.ranking.document import STOPWORDS

# Matches 42, 42.5, 1,200, 42%, 3x — the shapes that carry claims.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s*%?")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

# Symbol-shaped technical terms, matching the Stage 3 pattern vocabulary so
# both stages agree on what counts as a term.
_ACRONYM = re.compile(r"\b[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
_PREFIXED_SYMBOL = re.compile(r"\b[a-z]{1,4}[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")

#: Directional claims that must not be inverted. Checked both ways.
_DIRECTION_ANTONYMS = (
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("increases", "decreases"),
    ("higher", "lower"),
    ("greater", "smaller"),
    ("improve", "worsen"),
    ("improved", "worsened"),
    ("improves", "worsens"),
    ("more", "less"),
    ("gain", "loss"),
    ("upregulated", "downregulated"),
    ("positive", "negative"),
    ("effective", "ineffective"),
    ("significant", "insignificant"),
    ("faster", "slower"),
    ("reduced", "elevated"),
)

#: Claims a summary may only make if the abstract makes them first.
_OVERCLAIM_TERMS = frozenset(
    """
    first novel unprecedented breakthrough revolutionary proves proven proof
    cures cure definitive conclusively guarantees guaranteed always never
    best superior optimal groundbreaking landmark
    """.split()
)

#: Below this share of content words in common, the summary is about
#: something else.
#:
#: Measured rather than guessed, in two directions. Sweeping the labelled
#: set, 0.45 is the lowest value that still detects 100% of wrong-paper
#: cases (0.40 drops to 99.3%, 0.30 to 97.3%). Against live generation, it
#: raises first-attempt acceptance from 50% to 80% versus 0.55 - the
#: stricter value was rejecting genuine paraphrase, which cost a retry and
#: pushed 15% of papers to extractive text for no gain in drift detection.
#: See scripts/evaluate_grounding.py and scripts/evaluate_summaries.py.
_MIN_OVERLAP = 0.45

#: The prompt asks for 2-3 sentences; one extra is tolerated before it is
#: reported, because sentence splitting on scientific text is imprecise.
_MAX_SENTENCES = 4
_MIN_SENTENCES = 1


class GroundingChecker:
    """Verifies a summary against the abstract it claims to describe."""

    def __init__(
        self,
        *,
        min_overlap: float = _MIN_OVERLAP,
        max_sentences: int = _MAX_SENTENCES,
    ) -> None:
        self._min_overlap = min_overlap
        self._max_sentences = max_sentences

    def check(self, summary: str, abstract: str | None) -> GroundingVerdict:
        """Return a verdict, with one issue per distinct problem found.

        A paper with no abstract yields ``UNVERIFIABLE`` rather than a
        pass: there is nothing to check against, and reporting that
        honestly is the point of having a status enum at all.
        """
        if not abstract or not abstract.strip():
            return GroundingVerdict(
                status=GroundingStatus.UNVERIFIABLE,
                issues=[
                    GroundingIssue(
                        kind=IssueKind.LOW_OVERLAP,
                        detail="the paper has no abstract, so nothing can be verified",
                    )
                ],
            )

        summary = summary.strip()
        issues: list[GroundingIssue] = []
        issues.extend(self._check_numbers(summary, abstract))
        issues.extend(self._check_entities(summary, abstract))
        issues.extend(self._check_direction(summary, abstract))
        issues.extend(self._check_overclaims(summary, abstract))
        issues.extend(self._check_format(summary))

        overlap, overlap_issue = self._check_overlap(summary, abstract)
        if overlap_issue:
            issues.append(overlap_issue)

        return GroundingVerdict(
            status=GroundingStatus.UNGROUNDED if issues else GroundingStatus.GROUNDED,
            issues=issues,
            overlap=overlap,
        )

    # --- individual checks -------------------------------------------------

    @staticmethod
    def _check_numbers(summary: str, abstract: str) -> list[GroundingIssue]:
        """Every figure in the summary must occur in the abstract.

        Compared on the bare digits so that "42%", "42 %" and "42" agree,
        and thousands separators do not cause false alarms.
        """
        abstract_numbers = {_normalize_number(m) for m in _NUMBER.findall(abstract)}
        issues = []
        for match in _NUMBER.finditer(summary):
            value = _normalize_number(match.group(0))
            if not value or value in abstract_numbers:
                continue
            issues.append(
                GroundingIssue(
                    kind=IssueKind.FABRICATED_NUMBER,
                    detail=f"the summary states {match.group(0).strip()!r}, "
                    "which does not appear in the abstract",
                    span=match.group(0).strip(),
                )
            )
        return issues

    @staticmethod
    def _check_entities(summary: str, abstract: str) -> list[GroundingIssue]:
        """Named methods, genes and acronyms must come from the abstract."""
        lowered_abstract = abstract.lower()
        issues = []
        seen: set[str] = set()
        for pattern in (_ACRONYM, _PREFIXED_SYMBOL):
            for match in pattern.finditer(summary):
                term = match.group(0)
                key = term.lower()
                if key in seen or key in lowered_abstract:
                    continue
                seen.add(key)
                issues.append(
                    GroundingIssue(
                        kind=IssueKind.FABRICATED_ENTITY,
                        detail=f"the summary mentions {term!r}, "
                        "which the abstract never names",
                        span=term,
                    )
                )
        return issues

    def _check_overlap(
        self, summary: str, abstract: str
    ) -> tuple[float, GroundingIssue | None]:
        """Measure how much of the summary's vocabulary the abstract supports."""
        summary_words = _content_words(summary)
        if not summary_words:
            return 0.0, GroundingIssue(
                kind=IssueKind.LOW_OVERLAP, detail="the summary has no content words"
            )

        abstract_words = _content_words(abstract)
        shared = sum(1 for word in summary_words if word in abstract_words)
        overlap = shared / len(summary_words)
        if overlap >= self._min_overlap:
            return overlap, None
        return overlap, GroundingIssue(
            kind=IssueKind.LOW_OVERLAP,
            detail=f"only {overlap:.0%} of the summary's content words appear in the "
            f"abstract (needs {self._min_overlap:.0%})",
        )

    @staticmethod
    def _check_direction(summary: str, abstract: str) -> list[GroundingIssue]:
        """Catch a reversed direction of effect.

        Narrow by design: it fires only when the summary asserts one side
        of an antonym pair and the abstract asserts the other. That is the
        single paraphrase error most likely to actively mislead a
        researcher, and the one worth a targeted check.
        """
        summary_words = _content_words(summary)
        abstract_words = _content_words(abstract)
        issues = []
        for positive, negative in _DIRECTION_ANTONYMS:
            for stated, opposite in ((positive, negative), (negative, positive)):
                if (
                    stated in summary_words
                    and stated not in abstract_words
                    and opposite in abstract_words
                ):
                    issues.append(
                        GroundingIssue(
                            kind=IssueKind.CONTRADICTED_DIRECTION,
                            detail=f"the summary says {stated!r} where the abstract "
                            f"says {opposite!r}",
                            span=stated,
                        )
                    )
        return issues

    @staticmethod
    def _check_overclaims(summary: str, abstract: str) -> list[GroundingIssue]:
        """Novelty and certainty language the abstract does not itself use."""
        abstract_words = _content_words(abstract)
        issues = []
        for word in sorted(_content_words(summary) & _OVERCLAIM_TERMS):
            if word not in abstract_words:
                issues.append(
                    GroundingIssue(
                        kind=IssueKind.OVERCLAIM,
                        detail=f"the summary claims {word!r}, which the abstract "
                        "does not assert",
                        span=word,
                    )
                )
        return issues

    def _check_format(self, summary: str) -> list[GroundingIssue]:
        """The prompt asks for 2-3 sentences; enforce a tolerant bound."""
        if not summary:
            return [GroundingIssue(kind=IssueKind.FORMAT, detail="the summary is empty")]
        count = count_sentences(summary)
        if count > self._max_sentences:
            return [
                GroundingIssue(
                    kind=IssueKind.FORMAT,
                    detail=f"the summary is {count} sentences; at most "
                    f"{self._max_sentences} were requested",
                )
            ]
        if count < _MIN_SENTENCES:
            return [
                GroundingIssue(
                    kind=IssueKind.FORMAT, detail="the summary is not a complete sentence"
                )
            ]
        return []


def count_sentences(text: str) -> int:
    """Count sentences, tolerating the abbreviations abstracts are full of.

    Splitting on every period would count "et al." and "vs." as sentence
    boundaries, so a period only ends a sentence when followed by
    whitespace or the end of the string, and known abbreviations are
    excluded first.
    """
    cleaned = text
    for abbreviation in ("et al.", "e.g.", "i.e.", "vs.", "cf.", "Fig.", "approx."):
        cleaned = cleaned.replace(abbreviation, abbreviation.replace(".", ""))
    return len([part for part in _SENTENCE_END.split(cleaned) if part.strip()])


def _content_words(text: str) -> set[str]:
    """Lowercased words that carry meaning, for overlap comparison."""
    return {
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def _normalize_number(raw: str) -> str:
    """Reduce a numeric token to bare digits so formats compare equal."""
    return raw.replace(",", "").replace("%", "").replace(" ", "").rstrip(".")
