"""Hybrid relevance ranking.

Neither signal is sufficient on its own:

* **Semantic only** fails on the queries researchers actually type — a gene
  name, an assay, a model name, an acronym. A bi-encoder smooths rare
  tokens toward their neighbourhood, so ``PE3`` and ``PE2`` land in nearly
  the same place.
* **Lexical only** fails on paraphrase. A query for "cell-free DNA
  screening" misses a paper that only ever says "circulating tumour DNA".

So both are computed and fused. Which fusion wins is a measured question,
not an assumed one: all four strategies below are scored against the
golden set by ``scripts/evaluate_ranking.py``, and the README reports the
numbers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from enum import Enum

import numpy as np

from app.core.logging import get_logger
from app.models.paper import Paper, RelevanceScore
from app.services.ranking.document import document_text, tokenize
from app.services.ranking.embeddings import Embedder
from app.services.ranking.lexical import BM25Scorer

logger = get_logger(__name__)

#: Strength of the evidence prior; see ``_evidence_weights``. Measured on
#: the 20-query golden set, not chosen by taste.
EVIDENCE_PRIOR_K = 3.0


class FusionStrategy(str, Enum):
    """How the two signals are combined."""

    #: Cosine similarity only. A baseline, and the ablation for "is lexical earning its place".
    SEMANTIC = "semantic"
    #: BM25 only. The other baseline.
    LEXICAL = "lexical"
    #: Weighted sum of min-max normalized scores.
    LINEAR = "linear"
    #: Reciprocal rank fusion — combines ranks rather than scores.
    RRF = "rrf"


class HybridRanker:
    """Re-ranks a candidate set against a query.

    Stateless between calls apart from the shared embedder, so one instance
    is safe to reuse across concurrent requests.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        strategy: FusionStrategy = FusionStrategy.LINEAR,
        alpha: float = 0.6,
        rrf_k: int = 60,
        bigrams: bool = False,
        evidence_k: float = EVIDENCE_PRIOR_K,
    ) -> None:
        """
        Args:
            alpha: weight on the semantic score in ``LINEAR`` fusion; the
                lexical score gets ``1 - alpha``.
            rrf_k: the RRF damping constant. 60 is the value from the
                original Cormack et al. formulation and is not tuned here.
            bigrams: index adjacent token pairs in the lexical signal. See
                ``BM25Scorer`` — measured, and it does not pay off on
                average, so it is off.
            evidence_k: strength of the document-length prior; ``0``
                disables it, which is how the ablation row in
                ``evaluate_ranking`` is produced.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if evidence_k < 0.0:
            raise ValueError("evidence_k must not be negative")
        self._embedder = embedder
        self._strategy = strategy
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._bigrams = bigrams
        self._evidence_k = evidence_k

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    @property
    def strategy(self) -> FusionStrategy:
        return self._strategy

    async def arank(self, query: str, papers: Sequence[Paper]) -> list[Paper]:
        """Async wrapper — embedding is CPU-bound and would block the loop."""
        return await asyncio.to_thread(self.rank, query, papers)

    def rank(self, query: str, papers: Sequence[Paper]) -> list[Paper]:
        """Return the papers sorted by relevance, each carrying its score.

        Inputs are not mutated; every returned paper is a copy with
        ``score`` populated.
        """
        if not papers:
            return []

        semantic = self._semantic_scores(query, papers)
        lexical = BM25Scorer(papers, bigrams=self._bigrams).score(query)
        combined = self._fuse(semantic, lexical) * _evidence_weights(
            papers, self._evidence_k
        )

        order = np.argsort(-combined, kind="stable")
        ranked: list[Paper] = []
        for position, index in enumerate(order, start=1):
            paper = papers[index].model_copy(deep=True)
            paper.score = RelevanceScore(
                combined=float(np.clip(combined[index], 0.0, 1.0)),
                semantic=float(semantic[index]),
                lexical=float(lexical[index]),
                rank=position,
                strategy=self._strategy.value,
            )
            ranked.append(paper)
        return ranked

    # --- signals ---------------------------------------------------------

    def _semantic_scores(self, query: str, papers: Sequence[Paper]) -> np.ndarray:
        """Cosine similarity between the query and each paper.

        Both sides are L2-normalized by the embedder, so the dot product is
        the cosine. The query is embedded from the user's *raw* input —
        including a full pasted abstract — not from the keyword string that
        was sent to the upstream APIs, which is the whole reason the raw
        text is preserved by the query parser.
        """
        texts = [document_text(paper) for paper in papers]
        query_vector = self._embedder.encode([query])
        paper_vectors = self._embedder.encode(texts)
        if query_vector.size == 0 or paper_vectors.size == 0:
            return np.zeros(len(papers), dtype=np.float32)
        return np.asarray(paper_vectors @ query_vector[0], dtype=np.float32)

    # --- fusion ----------------------------------------------------------

    def _fuse(self, semantic: np.ndarray, lexical: np.ndarray) -> np.ndarray:
        match self._strategy:
            case FusionStrategy.SEMANTIC:
                return _minmax(semantic)
            case FusionStrategy.LEXICAL:
                return _minmax(lexical)
            case FusionStrategy.LINEAR:
                return self._alpha * _minmax(semantic) + (1 - self._alpha) * _minmax(lexical)
            case FusionStrategy.RRF:
                return _minmax(_rrf(semantic, lexical, k=self._rrf_k))
        raise ValueError(f"unhandled strategy: {self._strategy}")


def _evidence_weights(papers: Sequence[Paper], k: float) -> np.ndarray:
    """Discount candidates that have too little text to have earned a score.

    Both signals reward term density, and a record whose entire text is a
    two-word title is maximally dense by construction: every token it has
    is a query token. Crossref returns many such records — a bare title,
    no abstract — and they were arriving at rank 1 ahead of full papers.
    The clearest case: the query "federated learning differential privacy
    medical imaging" put a title-only record called simply *Differential
    privacy* first, with the highest cosine in the whole candidate set,
    above a dozen papers that actually do federated DP on medical images.

    The correction is a document prior, ``L / (L + k)``, on the token
    count. It is deliberately not a rule about missing abstracts: a paper
    with a two-sentence abstract has more evidence than a bare title and
    less than a full one, and a continuous weight says that, where a
    ``has_abstract`` flag cannot.

    **What the golden set actually showed.** Controlling for relevance
    grade, title-only records were ranking 15-31 percentile points above
    equally-relevant records that had abstracts — at *every* grade, so it
    was a scoring artefact rather than a real quality difference. (The
    pooled means hid this: title-only records are more relevant on
    average, grade 2.29 vs 2.11, because Crossref's stubs are often
    reviews. That coincidence is why the bug survived the 8-query set.)

    ``k`` was chosen by a paired bootstrap over the 20 queries, not by
    taking the best cell in a sweep. Only ``k`` in [2, 4] gives an
    improvement whose 95% interval excludes zero: at ``k=3``, NDCG@10
    +0.014 [+0.004, +0.025], 10 queries better and 3 worse. The larger
    apparent gains further out are noise — ``k=20`` scored NDCG@5 +0.038,
    which looks like the winner until the interval comes back as
    [-0.011, +0.105] on a 6-4 split. This is a small, real effect, and
    reporting it as a large one would be dishonest.

    The prior is applied to every fusion strategy, so the ablation rows
    stay ablations of the system that actually ships.
    """
    if k <= 0.0 or not papers:
        return np.ones(len(papers), dtype=np.float32)
    # Tokenized a second time here rather than reusing the BM25 corpus:
    # that corpus follows the bigram flag, which would silently double the
    # lengths and change what k means. A few dozen candidates make the
    # duplicated work irrelevant.
    lengths = np.array(
        [len(tokenize(document_text(paper))) for paper in papers], dtype=np.float32
    )
    return lengths / (lengths + k)


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Rescale to 0-1 within the candidate set.

    Necessary because BM25 is unbounded while cosine is roughly [-1, 1];
    a weighted sum of the raw values would be dominated by whichever
    happened to have the larger range. The cost is that absolute
    similarity is discarded — a candidate set where nothing matches still
    produces a 1.0 at the top. The raw components stay on
    ``RelevanceScore`` so that is visible rather than hidden.
    """
    if scores.size == 0:
        return scores.astype(np.float32)
    low = float(scores.min())
    high = float(scores.max())
    if high - low < 1e-9:
        # No discrimination available; say so rather than inventing an order.
        return np.full_like(scores, 0.0 if high <= 0 else 1.0, dtype=np.float32)
    return ((scores - low) / (high - low)).astype(np.float32)


def _rrf(*score_arrays: np.ndarray, k: int) -> np.ndarray:
    """Reciprocal rank fusion: sum of ``1 / (k + rank)`` across signals.

    Rank-based rather than score-based, so it is immune to the scale
    mismatch that forces min-max normalization on the linear strategy, and
    to outliers that would distort that normalization.
    """
    total = np.zeros(score_arrays[0].shape, dtype=np.float32)
    for scores in score_arrays:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(scores.shape, dtype=np.int64)
        ranks[order] = np.arange(1, scores.size + 1)
        total += (1.0 / (k + ranks)).astype(np.float32)
    return total
