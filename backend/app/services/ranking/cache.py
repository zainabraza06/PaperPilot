"""An LRU cache in front of an embedder.

Two stages of the pipeline embed the same papers during a single search:
the ranker scores them against the query, and the clusterer groups them.
Embedding a 24-paper result set costs roughly two seconds on CPU, so doing
it twice is the single most expensive avoidable thing the request does.

Wrapping the embedder rather than threading vectors through the call chain
keeps both consumers ignorant of each other: they ask for embeddings, and
the second one gets them for free. The cache also survives across
searches, so a paper that appears in two different result sets — common,
since researchers refine queries — is embedded once.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from app.core.logging import get_logger
from app.services.ranking.embeddings import Embedder

logger = get_logger(__name__)


class CachedEmbedder:
    """An ``Embedder`` that remembers what it has already encoded.

    Keyed by a digest of the text rather than the text itself, so memory is
    bounded by the entry count regardless of abstract length. Safe to share
    across threads: ``HybridRanker.arank`` runs encoding in a worker
    thread, so concurrent searches will call this simultaneously.
    """

    def __init__(self, inner: Embedder, capacity: int = 8192) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._inner = inner
        self._capacity = capacity
        self._entries: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for ``texts``, encoding only the cache misses.

        Misses are collected and encoded in one batch rather than one call
        each: batching is most of the throughput advantage of a transformer
        on CPU, and per-text calls would throw it away.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        keys = [_digest(text) for text in texts]
        vectors: list[np.ndarray | None] = []
        missing_indices: list[int] = []

        with self._lock:
            for position, key in enumerate(keys):
                cached = self._entries.get(key)
                if cached is None:
                    vectors.append(None)
                    missing_indices.append(position)
                    self.misses += 1
                else:
                    self._entries.move_to_end(key)
                    vectors.append(cached)
                    self.hits += 1

        if missing_indices:
            # Encoding happens outside the lock: it is the slow part, and
            # holding the lock through it would serialize concurrent searches.
            encoded = self._inner.encode([texts[i] for i in missing_indices])
            with self._lock:
                for offset, position in enumerate(missing_indices):
                    vector = np.asarray(encoded[offset], dtype=np.float32)
                    vectors[position] = vector
                    self._entries[keys[position]] = vector
                    self._entries.move_to_end(keys[position])
                while len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)

        return np.vstack([v for v in vectors if v is not None])

    def stats(self) -> dict[str, int]:
        """Hit/miss counters, for logging and the demo script."""
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "entries": len(self._entries)}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
