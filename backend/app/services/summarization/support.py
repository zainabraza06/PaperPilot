"""Sentence-level support checking — the answer to the lexical blind spot.

The lexical rules in :mod:`grounding` catch fabricated numbers, invented
entities, reversed directions and wholesale drift. What they provably
cannot catch, measured at 100% slip-through, is a **recombination**: a
fluent claim assembled entirely from the abstract's own vocabulary that
the abstract never actually makes. "The editing efficiency was caused by
the pegRNA design" uses only words that are present, invents no number,
and shares most of its content words with the source — every lexical rule
passes it.

The fix has to be inferential, and the textbook answer is an NLI model.
Two things argue against reaching for one here:

* It is a second model to download, load and keep resident, on top of the
  embedder that is already in memory.
* It is a model grading a model, which is the objection raised against
  LLM-as-judge in the first place — just with a smaller judge.

So this uses a weaker but honest proxy, built from the embedder that is
already loaded. **Every sentence of a summary should be a compression or
paraphrase of something the abstract actually says.** If a sentence's best
alignment to any single abstract sentence is poor, the sentence is
asserting something the source does not — even when every word in it came
from the source.

That last clause is the whole point. A recombination has high *vocabulary*
overlap and low *sentence-level* alignment, because it welds concepts from
two different sentences into a proposition that appears in neither. The
lexical checks look at the bag of words; this looks at whether any one
source sentence supports the claim.

**What this still does not do.** It is a similarity threshold, not
entailment. It cannot tell "A causes B" from "B causes A" when both
sentences discuss A and B together — the direction check covers the common
cases of that, and the rest is documented as remaining exposure rather
than papered over. The measured numbers are in the README, including where
it fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from app.services.ranking.embeddings import Embedder

#: Sentence boundaries, tolerating the abbreviations abstracts are full of.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "vs.", "cf.", "Fig.", "approx.", "ca.")

#: A sentence whose best alignment to any source window is below this is
#: not supported by the abstract.
#:
#: Chosen against *live generation*, not against synthetic paraphrase.
#:
#:              recombinations    false cautions on   cautions on *live*
#:   support        flagged      synthetic paraphrase   generated text
#:     0.50          61.3%              0.0%                 6.4%
#:     0.55          71.3%              0.2%                10.6%
#:     0.65          87.2%              0.5%                29.8%
#:     0.70          92.6%              0.6%                43.6%
#:     0.75          97.6%              1.1%                  --
#:
#: **Read the last two columns against each other.** They are measuring the
#: same thing - how often the check cries wolf - and they disagree by two
#: orders of magnitude. On hand-built paraphrases 0.70 costs 0.6% and looks
#: nearly free, which is what the synthetic benchmark alone would have
#: recommended. Against real generated text the same threshold cautions
#: 43.6%. The paraphrases were built by rewriting sentences; a model
#: summarizing an abstract compresses three sentences into one, and no
#: rewrite rule does that.
#:
#: So the operating point is 0.50, chosen on the live column. A caution
#: that fires on one summary in sixteen is worth reading; one that fires on
#: two in five is noise a reader learns to skip, which would cost the
#: signal entirely. Three fifths of recombinations flagged at that price is
#: the trade, and it is made against a baseline where none of them were
#: caught at all.
#:
#: (Synthetic numbers: 934 attacks and 664 paraphrases over 332 abstracts,
#: ``scripts/evaluate_grounding.py --sweep-support``. Live numbers: 94
#: generated summaries, ``scripts/evaluate_summaries.py --sweep-support``.)
_MIN_SUPPORT = 0.50

#: Sentences shorter than this carry no claim worth checking ("We show
#: this.", "Results were mixed."), and short text embeds unreliably.
_MIN_WORDS = 4

#: How many consecutive abstract sentences one summary sentence may be
#: compared against. Three covers the usual compression ratio; going wider
#: starts matching a summary sentence against half the abstract, which
#: would accept almost anything.
_MAX_WINDOW = 3


@dataclass(frozen=True)
class UnsupportedSentence:
    """One summary sentence with no adequate source sentence behind it."""

    text: str
    #: Best cosine similarity achieved against any abstract sentence.
    best_support: float
    #: The abstract sentence that came closest, for explaining the verdict.
    closest: str


class SemanticSupportChecker:
    """Checks each summary sentence against the abstract, sentence by sentence."""

    def __init__(
        self,
        embedder: Embedder,
        min_support: float = _MIN_SUPPORT,
        max_window: int = _MAX_WINDOW,
    ) -> None:
        self._embedder = embedder
        self._min_support = min_support
        self._max_window = max_window

    @property
    def min_support(self) -> float:
        return self._min_support

    def unsupported(self, summary: str, abstract: str) -> list[UnsupportedSentence]:
        """Return the summary sentences the abstract does not support.

        Embeds both sides once and compares every summary sentence against
        every abstract sentence — a summary is 2-3 sentences and an
        abstract a dozen, so the full matrix is trivial to compute and
        avoids the ordering assumptions an alignment heuristic would make.
        """
        summary_sentences = [
            sentence
            for sentence in split_sentences(summary)
            if len(sentence.split()) >= _MIN_WORDS
        ]
        abstract_sentences = split_sentences(abstract)
        if not summary_sentences or not abstract_sentences:
            return []

        # Compare against contiguous *windows*, not single sentences.
        #
        # This correction matters more than the threshold does. Summarizing
        # is compression: one good summary sentence routinely condenses two
        # or three consecutive abstract sentences, and scoring it against
        # each of them individually punishes exactly the behaviour the
        # feature exists to produce. Measured against live generation,
        # single-sentence comparison rejected 44% of legitimate summaries.
        #
        # A window is still contiguous, which is what keeps the check
        # meaningful: condensing adjacent material is normal summarization,
        # while welding together claims from opposite ends of an abstract
        # is the recombination this is looking for.
        windows = _contiguous_windows(abstract_sentences, self._max_window)

        summary_vectors = self._embedder.encode(summary_sentences)
        abstract_vectors = self._embedder.encode([text for text, _ in windows])
        if summary_vectors.size == 0 or abstract_vectors.size == 0:
            return []

        # Both sides are L2-normalized by the embedder contract, so the dot
        # product is the cosine.
        similarity = summary_vectors @ abstract_vectors.T
        best_indices = np.argmax(similarity, axis=1)
        best_scores = np.max(similarity, axis=1)

        failures: list[UnsupportedSentence] = []
        for position, sentence in enumerate(summary_sentences):
            score = float(best_scores[position])
            if score >= self._min_support:
                continue
            failures.append(
                UnsupportedSentence(
                    text=sentence,
                    best_support=score,
                    closest=windows[int(best_indices[position])][0],
                )
            )
        return failures


def _contiguous_windows(
    sentences: list[str], max_window: int
) -> list[tuple[str, tuple[int, int]]]:
    """Every run of 1..max_window consecutive sentences, with its span.

    Windows overlap deliberately: a summary sentence condensing sentences
    2 and 3 should match the (2, 3) window regardless of how sentence 2
    alone scores.
    """
    windows: list[tuple[str, tuple[int, int]]] = []
    for size in range(1, max_window + 1):
        for start in range(len(sentences) - size + 1):
            windows.append((" ".join(sentences[start : start + size]), (start, start + size)))
    return windows


def split_sentences(text: str) -> list[str]:
    """Split into sentences without breaking on common abbreviations.

    Abstracts are dense with "et al." and "e.g."; splitting on every period
    would shred them into fragments that embed meaninglessly.
    """
    guarded = text
    for abbreviation in _ABBREVIATIONS:
        guarded = guarded.replace(abbreviation, abbreviation.replace(".", "\x00"))
    parts = [
        part.replace("\x00", ".").strip()
        for part in _SENTENCE_SPLIT.split(guarded.strip())
    ]
    return [part for part in parts if part]
