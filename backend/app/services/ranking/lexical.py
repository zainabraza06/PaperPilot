"""Lexical relevance via BM25.

Semantic similarity alone loses exactly the queries researchers care most
about: a specific gene name, an assay, an acronym, a named model. Those are
rare tokens that a bi-encoder smooths away and that BM25 — which rewards
rare terms explicitly through IDF — scores correctly. Hence the hybrid.

**Scope note:** the BM25 index is built over the *candidate set returned by
the fan-out*, not over a global corpus. IDF is therefore relative to those
few dozen papers. That is the right choice here — we are re-ranking a
result set, not searching a collection — but it has two consequences worth
naming:

1. A term that is common *within the candidates* is discounted even if it
   is rare in the literature at large. For a query like "prime editing",
   every candidate mentions prime editing, so the phrase carries almost no
   weight and the ranking is decided by the incidental terms around it.
2. On a very small candidate set the signal collapses entirely. Okapi IDF
   is ``log((N - df + 0.5) / (df + 0.5))``, which is exactly 0 when a term
   appears in half the documents — so with two candidates every score is
   zero. Harmless in practice (a fan-out returns dozens) but it means the
   lexical signal cannot be relied on for tiny result sets, and the hybrid
   correctly falls back to the semantic one there.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.models.paper import Paper
from app.services.ranking.document import document_text, tokenize


class BM25Scorer:
    """Scores a query against a fixed candidate set with Okapi BM25."""

    def __init__(self, papers: Sequence[Paper], *, bigrams: bool = False) -> None:
        """
        Args:
            bigrams: also index adjacent token pairs, so multi-word terms
                match as units. Measured on the 20-query golden set: it
                fixes the query it was built for (prime-editing NDCG@10
                0.592 -> 0.689) but *lowers* the average (0.915 -> 0.898),
                because doubling the term space dilutes unigram IDF
                everywhere else. Off by default; kept because it is the
                right lever for a phrase-heavy corpus and the trade-off is
                measured, not assumed.

                The conclusion survived the golden set growing from 8
                queries to 20, which is worth noting because most of the
                other numbers in this project did not.
        """
        self._bigrams = bigrams
        self._corpus = [
            tokenize(document_text(paper), bigrams=bigrams) for paper in papers
        ]
        self._index = self._build_index(self._corpus)

    @staticmethod
    def _build_index(corpus: list[list[str]]) -> object | None:
        """Build the BM25 index, tolerating empty or all-empty corpora."""
        if not corpus or not any(corpus):
            # rank_bm25 divides by the average document length and raises
            # on a corpus of empty documents.
            return None
        from rank_bm25 import BM25Okapi

        index: object = BM25Okapi(corpus)
        return index

    def score(self, query: str) -> np.ndarray:
        """Return one raw BM25 score per candidate, in candidate order.

        Scores are unbounded and corpus-dependent; normalization is the
        fuser's job, not this class's.
        """
        if self._index is None or not self._corpus:
            return np.zeros(len(self._corpus), dtype=np.float32)
        tokens = tokenize(query, bigrams=self._bigrams)
        if not tokens:
            return np.zeros(len(self._corpus), dtype=np.float32)
        scores = self._index.get_scores(tokens)  # type: ignore[attr-defined]
        return np.asarray(scores, dtype=np.float32)
