"""Extractive summarization — the fallback that is grounded by construction.

Two jobs:

* **No provider configured.** The app still shows a summary for every
  paper, so cloning the repo without an API key gives a working product
  rather than an empty column.
* **Generation could not be grounded.** When a model's output fails the
  check twice, showing the failed text with a warning would be worse than
  showing something true. Sentences lifted verbatim from the abstract
  cannot hallucinate, because they are the abstract.

It is genuinely worse than a good generated summary — it cannot compress,
paraphrase, or lead with the finding — and it is always labelled
``SummaryOrigin.EXTRACTIVE`` so a reader knows which they are looking at.
"""

from __future__ import annotations

import re

from app.services.ranking.document import STOPWORDS

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")

#: Structured-abstract labels, stripped so a summary does not open with
#: "Background:". PubMed abstracts are re-assembled with these, and Crossref
#: JATS abstracts run them together with no colon at all - real records
#: begin "Abstract Motivation In single-cell RNA sequencing...".
#:
#: A label is only stripped when a colon follows, or when the next word is
#: capitalised. Without that guard an abstract opening "Results show that..."
#: would lose its subject and become "show that...".
#: The case-insensitive flag is scoped to the label group with ``(?i:...)``
#: rather than applied to the whole pattern. A global flag would also make
#: the ``[A-Z(]`` lookahead case-insensitive, which silently disables the
#: guard - "Results show that..." then loses its subject.
_SECTION_PREFIX = re.compile(
    r"^\s*(?i:abstract|background|objectives?|motivation|methods?|results?|conclusions?|"
    r"purpose|aims?|introduction|findings|interpretation|importance|significance|summary)"
    r"(?::\s*|\s+(?=[A-Z(]))"
)

#: Sentences opening a results or conclusion section carry the finding,
#: which is what a researcher scanning a list actually wants.
_FINDING_MARKERS = (
    "we found",
    "we show",
    "we demonstrate",
    "results",
    "here we",
    "conclusion",
    "we report",
    "our findings",
    "these results",
)


class ExtractiveSummarizer:
    """Selects sentences from the abstract itself."""

    model_id = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences

    def summarize(self, title: str, abstract: str | None) -> str:
        """Return 2-3 sentences drawn verbatim from ``abstract``.

        Selection is the abstract's opening sentence — which states the
        problem — plus the highest-scoring remaining sentences, where the
        score rewards overlap with the title and sentences that announce a
        finding. Chosen sentences are then emitted in their original order,
        so the result reads as prose rather than as a ranked list.
        """
        if not abstract or not abstract.strip():
            return title.strip()

        sentences = _split_sentences(abstract)
        if not sentences:
            return abstract.strip()
        if len(sentences) <= self._max_sentences:
            return " ".join(sentences)

        title_words = _content_words(title)
        scored: list[tuple[float, int]] = []
        for index, sentence in enumerate(sentences):
            words = _content_words(sentence)
            if not words:
                continue
            overlap = len(words & title_words) / max(len(title_words), 1)
            lowered = sentence.lower()
            finding = 1.0 if any(marker in lowered for marker in _FINDING_MARKERS) else 0.0
            # A mild positional bonus: abstracts front-load the question and
            # back-load the answer, so the middle is the least informative.
            position = 0.5 if index == 0 else (0.3 if index >= len(sentences) - 2 else 0.0)
            scored.append((overlap + finding + position, index))

        if not scored:
            return " ".join(sentences[: self._max_sentences])

        scored.sort(reverse=True)
        chosen = sorted(index for _, index in scored[: self._max_sentences])
        return " ".join(sentences[index] for index in chosen)


def _split_sentences(text: str) -> list[str]:
    """Split into sentences and drop structured-abstract section labels."""
    parts = []
    for raw in _SENTENCE.split(text.strip()):
        cleaned = raw.strip()
        # Repeatedly, because "Abstract Motivation In single-cell..." stacks
        # two labels before the sentence actually starts.
        while True:
            stripped = _SECTION_PREFIX.sub("", cleaned, count=1).strip()
            if stripped == cleaned:
                break
            cleaned = stripped
        if cleaned:
            parts.append(cleaned)
    return parts


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }
