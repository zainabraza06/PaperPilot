"""Recombination attacks, for evaluating the grounding checker.

Lives in the package rather than in the evaluation script because it is
the adversary the support checker is designed against, and the two should
be read together.

**Why this file is written carefully.** The checker and the attacks that
score it are both written here, so a high detection rate is only worth
something if the attacks are honestly hard. The earlier version generated
one formulaic sentence from the four most frequent words — trivially
detectable and worthless as a benchmark. These five families instead build
fluent claims out of the abstract's *own sentences*:

* ``invented_causation`` — two facts the abstract states separately,
  welded into a causal claim it never makes.
* ``swapped_roles`` — a real relation with subject and object reversed.
* ``scope_inflation`` — a specific finding generalized to "all" or "any".
* ``conflated_finding`` — one sentence's outcome attached to another
  sentence's condition.
* ``invented_comparison`` — two things the abstract mentions, asserted to
  have been compared.

Every one of them has high vocabulary overlap with the source, invents no
number, names no absent entity, and reverses no direction word — so every
lexical rule passes them by construction. That is the point: they isolate
exactly the capability the lexical rules do not have.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from app.services.summarization.support import split_sentences

#: Phrases that would make an attack trivially detectable by the overclaim
#: rule, defeating the purpose of the attack.
_BANNED = ("first", "prove", "novel", "best", "superior", "always", "never")

_DETERMINER = re.compile(r"^(the|a|an|our|their|these|this|those|its)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class Attack:
    """One generated claim, with the family that produced it."""

    text: str
    family: str


def _noun_phrase(sentence: str, rng: random.Random) -> str | None:
    """Pull a plausible noun phrase out of a sentence.

    Deliberately shallow — a dependency parse would be more accurate, but
    the attacks only need to be *fluent enough to be a fair test*, and a
    shallow phrase keeps this dependency-free and deterministic.
    """
    words = sentence.split()
    if len(words) < 6:
        return None
    # Take a 2-4 word window that does not start mid-punctuation, biased
    # away from the sentence opening where the subject is usually generic.
    start = rng.randrange(1, max(2, len(words) - 4))
    span = words[start : start + rng.choice((2, 3))]
    phrase = " ".join(span).strip(" ,;:.()")
    phrase = _DETERMINER.sub("", phrase)
    if len(phrase) < 6 or any(word in phrase.lower() for word in _BANNED):
        return None
    # A phrase ending in a preposition or conjunction reads as truncated.
    if phrase.lower().split()[-1] in {"of", "in", "to", "and", "or", "with", "for", "the", "a"}:
        return None
    return phrase


def build_attacks(abstract: str, rng: random.Random) -> list[Attack]:
    """Generate recombination attacks against one abstract.

    Returns fewer than five when the abstract is too short or too
    repetitive to draw distinct phrases from — a bad attack is worse than
    no attack, because it inflates the score with a case the checker was
    never really tested on.
    """
    sentences = [s for s in split_sentences(abstract) if len(s.split()) >= 8]
    if len(sentences) < 3:
        return []

    # Draw from sentences that are far apart: welding together adjacent
    # clauses often produces something the abstract does imply.
    first = sentences[0]
    middle = sentences[len(sentences) // 2]
    last = sentences[-1]

    attacks: list[Attack] = []

    subject = _noun_phrase(middle, rng)
    other = _noun_phrase(last, rng)
    condition = _noun_phrase(first, rng)

    if subject and other and subject.lower() != other.lower():
        attacks.append(
            Attack(
                f"The authors show that {subject} is caused by {other}.",
                "invented_causation",
            )
        )
        attacks.append(
            Attack(
                f"The authors report that {other} depends on {subject}, "
                f"rather than the other way around.",
                "swapped_roles",
            )
        )
        attacks.append(
            Attack(
                f"{subject.capitalize()} outperformed {other} across every "
                f"condition tested.",
                "invented_comparison",
            )
        )

    if subject and condition and subject.lower() != condition.lower():
        attacks.append(
            Attack(
                f"In all settings examined, {subject} was observed for {condition}.",
                "scope_inflation",
            )
        )
        attacks.append(
            Attack(
                f"The authors find that {condition} produced {subject} in "
                f"every replicate.",
                "conflated_finding",
            )
        )

    return attacks
