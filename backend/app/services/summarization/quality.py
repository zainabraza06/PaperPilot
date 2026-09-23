"""Reference-free quality measurement for generated summaries.

Groundedness answers "is this summary *false*". It says nothing about
whether the summary is any *good* — a summary reading "This paper studies
proteins." is perfectly grounded and perfectly useless, and every check in
:mod:`grounding` and :mod:`support` would pass it. That gap is what this
module measures.

**Why reference-free.** The standard approach is ROUGE against
human-written reference summaries. There are none here, and writing a few
hundred by hand would produce a reference set of exactly one annotator's
taste. So these are intrinsic metrics — properties of the (summary,
abstract) pair — and each one is reported next to the *same* metric
computed for baselines over the *same* abstracts. That comparison is the
measurement. A raw coverage of 0.62 means nothing on its own; 0.62 against
a lead-3 baseline's 0.71 means something specific.

**The metrics, and what each is for.**

``coverage``
    Recall of content. For every abstract sentence, its best alignment to
    any summary sentence; averaged. This is deliberately the mirror image
    of the support check: support asks whether everything in the summary
    came from the abstract (precision), coverage asks how much of the
    abstract survived into the summary (recall). Together they bracket the
    two ways a summary fails — inventing and omitting.

``compression``
    Summary words over abstract words. Not a quality score by itself; it
    is the denominator the others have to be read against, because
    coverage is trivially bought by writing a longer summary.

``novelty``
    Share of summary bigrams that do not occur in the abstract. Separates
    abstractive summarization from extraction with the serial numbers
    filed off. Near zero means the model is copying; the extractive
    baseline scores near zero *by construction*, which is what makes it
    the useful reference point.

``longest_copied_span``
    The longest verbatim token run shared with the abstract, as a fraction
    of the summary. ``novelty`` can look healthy while one long clause is
    lifted whole; this catches that specific case.

``lead_bias``
    Where in the abstract the summary's content comes from, as a mean
    normalized position. Abstracts front-load the problem and back-load
    the finding, so a summary drawn only from the opening has missed the
    result. 0.5 is even coverage of the abstract; below ~0.35 means the
    model mostly rewrote the first few sentences.

``redundancy``
    Highest similarity between any two sentences of the summary. In three
    sentences there is no room to say the same thing twice.

**What this is not.** None of these measure factual correctness, and none
of them measure whether a human would find the summary useful. They are
proxies with known failure modes: a summary can score well on all six and
still be a bad summary, and a genuinely excellent terse summary will score
low on coverage. They are reported as a comparison against baselines
precisely because the absolute values are not meaningful.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from app.services.ranking.embeddings import Embedder
from app.services.summarization.support import split_sentences

_WORD = re.compile(r"[a-z0-9][a-z0-9\-']*")

#: Sentences below this length are fragments and embed unreliably; they are
#: excluded from the sentence-level metrics but still counted as words.
_MIN_WORDS = 4


@dataclass(frozen=True)
class QualityScores:
    """Intrinsic quality metrics for one (summary, abstract) pair."""

    coverage: float
    compression: float
    novelty: float
    longest_copied_span: float
    lead_bias: float
    redundancy: float

    @staticmethod
    def fields() -> tuple[str, ...]:
        """Metric names, in the order they should be reported."""
        return (
            "coverage",
            "compression",
            "novelty",
            "longest_copied_span",
            "lead_bias",
            "redundancy",
        )

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.fields()}


class QualityScorer:
    """Scores summaries against their source abstracts.

    Reuses the ranking embedder rather than loading anything new, for the
    same reason :mod:`support` does: the model is already resident, and a
    second one would be a second thing to ship, cache and keep in sync.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def score(self, summary: str, abstract: str) -> QualityScores | None:
        """Score one pair, or ``None`` if there is not enough text to score.

        Returning ``None`` rather than zeros matters: a pair that could not
        be scored must be excluded from an average, not averaged in as if
        it had scored badly.
        """
        summary_sentences = [
            sentence
            for sentence in split_sentences(summary)
            if len(sentence.split()) >= _MIN_WORDS
        ]
        abstract_sentences = [
            sentence
            for sentence in split_sentences(abstract)
            if len(sentence.split()) >= _MIN_WORDS
        ]
        if not summary_sentences or not abstract_sentences:
            return None

        summary_tokens = _tokenize(summary)
        abstract_tokens = _tokenize(abstract)
        if not summary_tokens or not abstract_tokens:
            return None

        summary_vectors = self._embedder.encode(summary_sentences)
        abstract_vectors = self._embedder.encode(abstract_sentences)
        if summary_vectors.size == 0 or abstract_vectors.size == 0:
            return None

        # Embedder contract guarantees L2-normalized rows, so the dot
        # product is the cosine. Rows are abstract sentences, columns are
        # summary sentences.
        similarity = abstract_vectors @ summary_vectors.T

        return QualityScores(
            coverage=float(np.mean(np.max(similarity, axis=1))),
            compression=len(summary_tokens) / len(abstract_tokens),
            novelty=_novelty(summary_tokens, abstract_tokens),
            longest_copied_span=_longest_copied_span(summary_tokens, abstract_tokens),
            lead_bias=_lead_bias(similarity),
            redundancy=_redundancy(summary_vectors),
        )


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bigrams(tokens: Sequence[str]) -> set[str]:
    return {f"{a} {b}" for a, b in pairwise(tokens)}


def _novelty(summary_tokens: list[str], abstract_tokens: list[str]) -> float:
    """Share of summary bigrams that do not appear in the abstract.

    Bigrams rather than unigrams: a summary of an abstract is *supposed*
    to reuse its vocabulary — the entities and methods have names — so
    unigram novelty would mostly measure how many synonyms the model
    reached for, which is not the question. A novel bigram means the words
    were recombined into a new phrase, which is what rewriting is.
    """
    summary_bigrams = _bigrams(summary_tokens)
    if not summary_bigrams:
        return 0.0
    source = _bigrams(abstract_tokens)
    return len(summary_bigrams - source) / len(summary_bigrams)


def _longest_copied_span(summary_tokens: list[str], abstract_tokens: list[str]) -> float:
    """Longest verbatim run shared with the abstract, over summary length.

    Classic longest-common-substring by dynamic programming, over tokens
    rather than characters. Abstracts and summaries are both short enough
    that the quadratic table is irrelevant — a few hundred by a few dozen.
    """
    if not summary_tokens or not abstract_tokens:
        return 0.0
    previous = [0] * (len(abstract_tokens) + 1)
    best = 0
    for summary_token in summary_tokens:
        current = [0] * (len(abstract_tokens) + 1)
        for column, abstract_token in enumerate(abstract_tokens, start=1):
            if summary_token == abstract_token:
                current[column] = previous[column - 1] + 1
                best = max(best, current[column])
        previous = current
    return best / len(summary_tokens)


def _lead_bias(similarity: np.ndarray) -> float:
    """Mean normalized position of the abstract sentences the summary used.

    For each *summary* sentence, find which abstract sentence it aligns to
    best, and take that sentence's position scaled to 0-1. Averaging those
    says where in the abstract the summary was sourced from. A single
    abstract sentence has no position to speak of, so it scores 0.5 —
    neutral — rather than 0.
    """
    rows = similarity.shape[0]
    if rows <= 1:
        return 0.5
    best_source = np.argmax(similarity, axis=0)
    return float(np.mean(best_source / (rows - 1)))


def _redundancy(summary_vectors: np.ndarray) -> float:
    """Highest cosine between any two distinct summary sentences."""
    if summary_vectors.shape[0] < 2:
        return 0.0
    similarity = summary_vectors @ summary_vectors.T
    np.fill_diagonal(similarity, -1.0)
    return float(np.max(similarity))
