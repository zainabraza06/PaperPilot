"""Tests for the semantic support check and the severity split.

These cover the part of the grounding layer that decides whether a
summary is *replaced* or merely *annotated*, which nothing else tested.
The offline harness in ``scripts/evaluate_grounding.py`` measures how
often the check is right; these pin what it does with the answer.

Similarity comes from a stub embedder rather than the real model, so the
assertions are about wiring and policy — which issues are produced, what
severity they carry, and whether they block — and not about a downloaded
model's numbers. Detection *rates* belong in the harness, where they are
measured against 934 attacks rather than asserted against one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from app.models.summary import GroundingStatus, IssueKind, IssueSeverity
from app.services.summarization.grounding import GroundingChecker
from app.services.summarization.support import SemanticSupportChecker, split_sentences

ABSTRACT = (
    "Prime editing enables precise genome edits without double-strand breaks. "
    "We benchmarked PE2 and PE3 across six primary human cell types using pegRNAs. "
    "Editing efficiency increased to 42% in T cells, with indel rates below 1.5%."
)

#: A claim built only from the abstract's own words that it never makes.
#: No invented number, no absent entity, no reversed direction — so every
#: lexical rule passes it by construction.
RECOMBINATION = (
    "The pegRNAs caused editing efficiency to increase to 42% in T cells. "
    "Prime editing without double-strand breaks produced indel rates below 1.5%."
)


class MarkedEmbedder:
    """Embeds by marker token, so cosine similarity is exactly controlled.

    A text containing markers gets those markers' axes, so two texts
    sharing a marker are similar and a text carrying two markers sits
    halfway between them. A text containing *no* marker is given an axis of
    its own, which makes it orthogonal to everything else — that is how
    "nothing in the abstract supports this" is expressed. Giving unmarked
    texts a *shared* axis instead would make them all identical to each
    other, which silently turns every such assertion into a tautology.
    """

    dimension = 64
    model_id = "marked"

    def __init__(self, markers: Sequence[str]) -> None:
        self._markers = list(markers)
        self._private: dict[str, int] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for index, marker in enumerate(self._markers):
                if marker in text:
                    vector[index] = 1.0
            if not vector.any():
                axis = self._private.setdefault(
                    text, len(self._markers) + len(self._private)
                )
                assert axis < self.dimension, "stub ran out of axes"
                vector[axis] = 1.0
            rows.append(vector)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


# --- the support check itself -------------------------------------------


def test_a_sentence_matching_a_source_sentence_is_supported() -> None:
    embedder = MarkedEmbedder(["benchmarked"])
    support = SemanticSupportChecker(embedder, min_support=0.5)
    assert support.unsupported("We benchmarked the method.", ABSTRACT) == []


def test_a_sentence_matching_nothing_is_reported_with_its_best_attempt() -> None:
    embedder = MarkedEmbedder(["benchmarked"])
    support = SemanticSupportChecker(embedder, min_support=0.5)
    failures = support.unsupported("Mitochondrial transport was unaffected here.", ABSTRACT)
    assert len(failures) == 1
    # The closest source text travels with the failure so a reader can be
    # shown *why* the sentence was flagged, not just that it was.
    assert failures[0].best_support < 0.5
    assert failures[0].closest in ABSTRACT or failures[0].closest


def test_short_sentences_carry_no_claim_and_are_skipped() -> None:
    """"We show this." asserts nothing, and short text embeds unreliably."""
    embedder = MarkedEmbedder(["benchmarked"])
    support = SemanticSupportChecker(embedder, min_support=0.99)
    assert support.unsupported("We show this.", ABSTRACT) == []


def test_a_summary_sentence_may_condense_consecutive_source_sentences() -> None:
    """The windowing behaviour, which is the whole reason this is usable.

    Summarizing is compression: one sentence routinely condenses two or
    three consecutive abstract sentences. Scoring against single sentences
    rejected 44% of legitimate summaries, so windows are contiguous runs.
    """
    # The summary sentence carries both markers, so it half-matches either
    # source sentence alone and fully matches the window containing both.
    embedder = MarkedEmbedder(["alpha", "beta"])
    abstract = "Findings about alpha. Findings about beta. An unrelated remark."
    summary = "Findings about alpha and about beta together."

    windowed = SemanticSupportChecker(embedder, min_support=0.9, max_window=3)
    single = SemanticSupportChecker(embedder, min_support=0.9, max_window=1)

    assert windowed.unsupported(summary, abstract) == []
    assert single.unsupported(summary, abstract), "single-sentence scoring rejects compression"


def test_windows_do_not_span_the_whole_abstract() -> None:
    """Contiguity is what keeps the check meaningful.

    Condensing adjacent material is normal summarization; welding together
    claims from opposite ends of an abstract is the recombination this
    exists to find. A window wide enough to cover both would accept
    anything.
    """
    embedder = MarkedEmbedder(["alpha", "omega"])
    abstract = "About alpha. A first filler sentence. A second filler sentence. About omega."
    summary = "Claims joining alpha and omega into one finding."
    support = SemanticSupportChecker(embedder, min_support=0.9, max_window=3)
    assert support.unsupported(summary, abstract)


def test_an_empty_side_is_not_treated_as_unsupported() -> None:
    support = SemanticSupportChecker(MarkedEmbedder(["x"]), min_support=0.5)
    assert support.unsupported("", ABSTRACT) == []
    assert support.unsupported("A reasonable summary sentence here.", "") == []


def test_sentence_splitting_does_not_break_on_abbreviations() -> None:
    """Abstracts are dense with "et al." and "e.g.", which naive splitting shreds."""
    sentences = split_sentences("Smith et al. reported a gain. We saw e.g. two cases.")
    assert len(sentences) == 2


# --- severity: what the verdict does with the answer ---------------------


def test_an_unsupported_claim_is_advisory_and_does_not_reject_the_summary() -> None:
    """The policy decision, pinned.

    At a threshold strict enough to catch most recombinations the check
    cautions ~44% of perfectly good summaries, so blocking on it would
    destroy more good summaries than it saves bad ones. It annotates, and
    the summary stands.
    """
    support = SemanticSupportChecker(MarkedEmbedder(["nothing-matches"]), min_support=0.9)
    verdict = GroundingChecker(support=support).check(RECOMBINATION, ABSTRACT)

    assert verdict.advisories
    assert all(i.kind is IssueKind.UNSUPPORTED_CLAIM for i in verdict.advisories)
    assert not verdict.blocking
    assert verdict.passed
    assert verdict.status is GroundingStatus.GROUNDED


def test_a_lexical_failure_still_blocks_even_alongside_an_advisory() -> None:
    support = SemanticSupportChecker(MarkedEmbedder(["nothing-matches"]), min_support=0.9)
    verdict = GroundingChecker(support=support).check(
        "The method reached 99.7% efficiency, which proves it is the first of its kind.",
        ABSTRACT,
    )
    assert verdict.blocking
    assert not verdict.passed
    assert verdict.status is GroundingStatus.UNGROUNDED


def test_blocking_and_advisories_partition_every_issue() -> None:
    support = SemanticSupportChecker(MarkedEmbedder(["nothing-matches"]), min_support=0.9)
    verdict = GroundingChecker(support=support).check(
        "The method reached 99.7% efficiency, which proves it is the first of its kind.",
        ABSTRACT,
    )
    assert len(verdict.blocking) + len(verdict.advisories) == len(verdict.issues)


def test_every_lexical_issue_blocks_by_default() -> None:
    """Severity defaults to blocking, so a new rule cannot be silently advisory."""
    verdict = GroundingChecker().check(
        "The method reached 99.7% efficiency, which proves it is the first of its kind.",
        ABSTRACT,
    )
    assert verdict.issues
    assert all(i.severity is IssueSeverity.BLOCKING for i in verdict.issues)


# --- the composition that actually ships --------------------------------


def test_without_a_support_check_the_recombination_passes_unremarked() -> None:
    """The baseline the support check was written against.

    Measured over 934 attacks, the lexical rules alone caught zero. This
    pins that the blind spot is a property of the lexical rules and not an
    accident of one example.
    """
    verdict = GroundingChecker().check(RECOMBINATION, ABSTRACT)
    assert verdict.passed
    assert not verdict.issues


def test_with_a_support_check_the_same_recombination_is_flagged() -> None:
    """...and that attaching the check is what changes the outcome.

    The summary is unchanged; only the checker's configuration differs.
    The result is a caution rather than a rejection, which is the shipped
    policy.
    """
    support = SemanticSupportChecker(MarkedEmbedder(["nothing-matches"]), min_support=0.9)
    verdict = GroundingChecker(support=support).check(RECOMBINATION, ABSTRACT)
    assert verdict.passed, "still accepted - the caution does not reject"
    assert verdict.advisories, "but no longer silent"


@pytest.mark.parametrize(
    ("min_support", "expect_flagged"),
    [(0.0, False), (0.5, False), (1.01, True)],
)
def test_the_threshold_is_what_decides(min_support: float, expect_flagged: bool) -> None:
    """The same sentence and abstract; only the threshold moves.

    A threshold of 0 flags nothing, and one above the maximum achievable
    cosine flags even an exact match. Nothing is special-cased at the ends.
    """
    support = SemanticSupportChecker(MarkedEmbedder(["benchmarked"]), min_support=min_support)
    failures = support.unsupported("We benchmarked the method thoroughly.", ABSTRACT)
    assert bool(failures) is expect_flagged
