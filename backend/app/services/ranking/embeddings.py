"""Text embedding for semantic relevance.

Two implementations behind one protocol:

* ``SentenceTransformerEmbedder`` — the real one, a bi-encoder from
  sentence-transformers. Loading it costs seconds and ~90 MB, so it is
  loaded lazily and shared process-wide.
* ``HashingEmbedder`` — a deterministic, dependency-free fallback. It
  exists so the test suite stays offline and fast, and so the app still
  ranks (worse, but predictably worse) on a machine that cannot download
  the model. Which one produced a ranking is always reported, because a
  quality number measured with the fallback would be misleading.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalized vectors.

    Normalization is part of the contract so that callers can use a plain
    dot product as cosine similarity.
    """

    @property
    def dimension(self) -> int:
        """Output vector width."""
        ...

    @property
    def model_id(self) -> str:
        """Short identifier, reported alongside any evaluation result."""
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an ``(len(texts), dimension)`` float array."""
        ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers bi-encoder.

    ``all-MiniLM-L6-v2`` is the default: 384 dimensions, fast enough to
    embed a 60-paper result set on CPU within a search request, and
    trained on a mixture that includes scientific text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_id = model_name
        self._model_name = model_name
        self._model: object | None = None
        # Loading is not thread-safe and can be triggered concurrently by
        # two in-flight searches on the first request after startup.
        self._lock = threading.Lock()
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dimension is not None
        return self._dimension

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # another thread won the race
                return
            # Imported here, not at module scope: importing torch costs
            # seconds, and the API should start without paying that.
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self._model_name)
            model = SentenceTransformer(self._model_name)
            dimension = model.get_sentence_embedding_dimension()
            if dimension is None:
                raise RuntimeError(f"{self._model_name} reported no embedding dimension")
            self._dimension = int(dimension)
            self._model = model
            logger.info("embedding model ready (dim=%d)", self._dimension)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._model
        assert model is not None
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """A deterministic bag-of-words embedder used as a fallback.

    Tokens are hashed into a fixed number of buckets with a signed hash
    (which cancels some collisions rather than compounding them) and
    weighted by square-root term frequency. It captures lexical overlap
    only — no synonymy, no word order — so it is genuinely worse than a
    trained model. It is here for offline tests and graceful degradation,
    never as a silent substitute: ``model_id`` says which one ran.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.model_id = f"hashing-{dimension}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in _TOKEN.findall(text.lower()):
                if len(token) > 2:
                    counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                digest = hashlib.md5(token.encode()).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, bucket] += sign * float(np.sqrt(count))
        return _l2_normalize(vectors)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows alone."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)


def build_embedder(model_name: str | None) -> Embedder:
    """Return the configured embedder, falling back if it cannot be loaded.

    A missing model must not take the search endpoint down: the fallback
    still produces a usable lexical-ish ranking, and the degradation is
    reported rather than hidden.
    """
    if not model_name:
        return HashingEmbedder()
    embedder = SentenceTransformerEmbedder(model_name)
    try:
        embedder._ensure_loaded()
    except Exception as exc:
        logger.warning(
            "could not load embedding model %r (%s); falling back to hashing embedder",
            model_name,
            exc,
        )
        return HashingEmbedder()
    return embedder
