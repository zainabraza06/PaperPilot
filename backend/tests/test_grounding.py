"""Grounding-check tests.

This is the module that decides whether a researcher is shown a claim the
abstract does not support, so the tests are written as a specification of
what must be caught and — equally important — what must *not* trigger a
false alarm. A checker that rejects good summaries is not "safe"; it just
degrades every result to extractive text.
"""

from __future__ import annotations

import pytest

from app.models.summary import GroundingStatus, IssueKind
from app.services.summarization.grounding import GroundingChecker, count_sentences

ABSTRACT = (
    "Prime editing enables precise genome edits without double-strand breaks. "
    "We benchmarked PE2 and PE3 across six primary human cell types using pegRNAs. "
    "Editing efficiency increased to 42% in T cells, with indel rates below 1.5%."
)


@pytest.fixture
def checker() -> GroundingChecker:
    return GroundingChecker()


def kinds(checker: GroundingChecker, summary: str, abstract: str = ABSTRACT) -> set[IssueKind]:
    return {issue.kind for issue in checker.check(summary, abstract).issues}


# --- summaries that must pass --------------------------------------------


def test_a_faithful_summary_passes(checker: GroundingChecker) -> None:
    verdict = checker.check(
        "The authors benchmarked PE2 and PE3 across six primary human cell types. "
        "Editing efficiency increased to 42% in T cells, with indel rates below 1.5%.",
        ABSTRACT,
    )
    assert verdict.status is GroundingStatus.GROUNDED
    assert verdict.issues == []
    assert verdict.overlap > 0.8


def test_paraphrase_that_stays_within_the_abstract_passes(checker: GroundingChecker) -> None:
    # The check must not demand verbatim copying, or generation is pointless.
    verdict = checker.check(
        "Prime editing achieves precise genome edits without double-strand breaks. "
        "Across six primary human cell types, efficiency increased to 42% in T cells.",
        ABSTRACT,
    )
    assert verdict.passed


def test_numbers_written_differently_still_match(checker: GroundingChecker) -> None:
    # "42 %" and "1,5" style variation must not read as fabrication.
    assert IssueKind.FABRICATED_NUMBER not in kinds(
        checker, "Efficiency increased to 42 % in T cells across six cell types."
    )


# --- fabrication ----------------------------------------------------------


def test_an_invented_number_is_caught(checker: GroundingChecker) -> None:
    # The highest-impact hallucination in a research tool.
    assert IssueKind.FABRICATED_NUMBER in kinds(
        checker, "Editing efficiency increased to 87% in T cells across six cell types."
    )


def test_an_invented_gene_or_method_is_caught(checker: GroundingChecker) -> None:
    assert IssueKind.FABRICATED_ENTITY in kinds(
        checker, "The authors used CRISPRoff and PE2 across six primary human cell types."
    )


def test_the_offending_text_is_reported_not_just_the_failure(
    checker: GroundingChecker,
) -> None:
    # The retry prompt quotes these back, and the UI can highlight them.
    verdict = checker.check(
        "Editing efficiency reached 87% using CRISPRoff in six human cell types.", ABSTRACT
    )
    spans = {issue.span for issue in verdict.issues}
    assert "87%" in spans
    assert "CRISPRoff" in spans


# --- drift and contradiction ---------------------------------------------


def test_a_summary_about_a_different_paper_is_caught(checker: GroundingChecker) -> None:
    verdict = checker.check(
        "This study examines photovoltaic conversion in perovskite solar cells "
        "under prolonged irradiation.",
        ABSTRACT,
    )
    assert IssueKind.LOW_OVERLAP in {issue.kind for issue in verdict.issues}
    assert verdict.overlap < 0.5


def test_a_reversed_direction_of_effect_is_caught(checker: GroundingChecker) -> None:
    # The one paraphrase error that actively misleads rather than just
    # being vague: the abstract says efficiency increased.
    assert IssueKind.CONTRADICTED_DIRECTION in kinds(
        checker,
        "Editing efficiency decreased to 42% in T cells across six primary human cell types.",
    )


def test_direction_words_the_abstract_itself_uses_are_fine(
    checker: GroundingChecker,
) -> None:
    assert IssueKind.CONTRADICTED_DIRECTION not in kinds(
        checker,
        "Efficiency increased to 42% in T cells, with indel rates below 1.5% using pegRNAs.",
    )


def test_novelty_and_certainty_claims_are_caught(checker: GroundingChecker) -> None:
    assert IssueKind.OVERCLAIM in kinds(
        checker,
        "This is the first study to prove prime editing works, benchmarking PE2 and PE3 "
        "across six primary human cell types.",
    )


def test_overclaim_words_the_abstract_uses_are_allowed() -> None:
    abstract = "We present the first genome-wide screen. It proves the mechanism."
    checker = GroundingChecker(min_overlap=0.3)
    assert IssueKind.OVERCLAIM not in kinds(
        checker, "The authors present the first genome-wide screen that proves the mechanism.", abstract
    )


# --- format ---------------------------------------------------------------


def test_an_overlong_summary_is_flagged(checker: GroundingChecker) -> None:
    long_summary = " ".join(
        [
            "Prime editing enables precise genome edits.",
            "We benchmarked PE2 and PE3.",
            "Six primary human cell types were used.",
            "Editing efficiency increased to 42%.",
            "Indel rates stayed below 1.5%.",
        ]
    )
    assert IssueKind.FORMAT in kinds(checker, long_summary)


def test_an_empty_summary_is_flagged(checker: GroundingChecker) -> None:
    assert IssueKind.FORMAT in kinds(checker, "")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One sentence.", 1),
        ("One. Two.", 2),
        ("Smith et al. reported this. Then more.", 2),  # "et al." is not a boundary
        ("Compare e.g. this vs. that. Done.", 2),
        ("No trailing period", 1),
    ],
)
def test_sentence_counting_tolerates_abbreviations(text: str, expected: int) -> None:
    assert count_sentences(text) == expected


# --- edge cases -----------------------------------------------------------


def test_a_paper_with_no_abstract_is_unverifiable_not_grounded(
    checker: GroundingChecker,
) -> None:
    # The distinction matters: "we could not check" must never be presented
    # as "we checked and it passed".
    verdict = checker.check("Some plausible summary text.", None)
    assert verdict.status is GroundingStatus.UNVERIFIABLE
    assert not verdict.passed


def test_an_empty_abstract_is_also_unverifiable(checker: GroundingChecker) -> None:
    assert checker.check("Anything", "   ").status is GroundingStatus.UNVERIFIABLE


def test_every_distinct_problem_is_reported_not_just_the_first(
    checker: GroundingChecker,
) -> None:
    # The retry prompt is only as good as the list of failures it quotes.
    verdict = checker.check(
        "This first-ever work proves that efficiency decreased to 87% using CRISPRoff.",
        ABSTRACT,
    )
    found = {issue.kind for issue in verdict.issues}
    assert IssueKind.FABRICATED_NUMBER in found
    assert IssueKind.FABRICATED_ENTITY in found
    assert IssueKind.OVERCLAIM in found


def test_the_overlap_threshold_is_configurable() -> None:
    summary = "Photovoltaic perovskite irradiation degradation was measured."
    assert not GroundingChecker(min_overlap=0.9).check(summary, ABSTRACT).passed
    # At zero, overlap can no longer be the reason for a rejection.
    lenient = GroundingChecker(min_overlap=0.0).check(summary, ABSTRACT)
    assert IssueKind.LOW_OVERLAP not in {issue.kind for issue in lenient.issues}


def test_the_lexical_rules_alone_are_blind_to_recombination(
    checker: GroundingChecker,
) -> None:
    """Pins the limitation these six rules have, so it is never overclaimed.

    A summary built entirely from the abstract's own vocabulary, asserting
    something the abstract never said, passes every rule in this module.
    Measured over 934 such attacks, the lexical rules caught zero.

    This is scoped to the *lexical* checker on purpose. The shipped
    configuration attaches a semantic support check that catches ~61% of
    these and is exercised in ``tests/test_support.py``; it is injected
    rather than built in, so this module stays testable with no model
    loaded. Neither closes the gap — the support check is a similarity
    threshold, not entailment.
    """
    misleading = (
        "PE3 increased editing efficiency in T cells to 42%, and pegRNAs caused "
        "indel rates below 1.5% in six primary human cell types."
    )
    assert checker.check(misleading, ABSTRACT).passed
