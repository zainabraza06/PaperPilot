"""How a ``Paper`` becomes text for ranking.

Both scorers — lexical and semantic — must see the *same* view of a paper,
or their scores are not comparable and fusing them is meaningless. That
view is defined once, here.
"""

from __future__ import annotations

import re
from itertools import pairwise

from app.models.paper import Paper

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")

# Deliberately minimal: BM25's IDF term already discounts words that appear
# everywhere in the candidate set, so an aggressive stoplist mostly removes
# signal. These are the closed-class words that carry none.
STOPWORDS = frozenset(
    """
    a an and are as at be been by for from had has have in into is it its of on or
    that the their there these this to was were what when which who will with
    """.split()
)


def document_text(paper: Paper) -> str:
    """Return the text used to represent a paper for both scorers.

    The title is repeated once. This is a deliberate, mild weighting: a
    term in the title is stronger evidence of aboutness than the same term
    buried in an abstract, and duplicating it is the cheapest way to say so
    to BM25 without inventing a separate field-weighting scheme.

    Papers with no abstract — around half of Crossref's records — fall back
    to their keywords and venue. That is genuinely weaker evidence, and it
    is a known limitation of ranking such records rather than a bug.
    """
    parts = [paper.title, paper.title]
    if paper.abstract:
        parts.append(paper.abstract)
    else:
        parts.extend(paper.keywords)
        if paper.journal:
            parts.append(paper.journal)
    return " ".join(part for part in parts if part)


def tokenize(text: str, *, bigrams: bool = False) -> list[str]:
    """Lowercase alphanumeric tokens, minus closed-class stopwords.

    With ``bigrams=True``, adjacent token pairs are appended as ``a_b``
    terms. This matters more than it looks: BM25 is a bag of words, so
    "prime editing" is scored as two independent terms, and a paper
    matching *editing* + *efficiency* + *CRISPR* + *human* outranks one
    that is actually about prime editing. Indexing the pair as a unit lets
    the lexical signal reward the phrase rather than its parts.
    """
    tokens = [
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]
    if not bigrams:
        return tokens
    return tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
