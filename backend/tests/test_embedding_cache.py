"""Tests for the embedding cache.

The cache is the reason ranking and clustering do not each pay for a
forward pass over the same papers, so "the second consumer gets a hit" is
the property that matters, and "the returned vectors are still correct and
in order" is the one that would be catastrophic to get wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from app.services.ranking.cache import CachedEmbedder


class CountingEmbedder:
    """Records every batch it is asked to encode."""

    dimension = 3
    model_id = "counting"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        # Deterministic and distinct per text, so ordering bugs show up.
        return np.asarray(
            [[float(len(t)), float(sum(map(ord, t)) % 97), 1.0] for t in texts],
            dtype=np.float32,
        )

    @property
    def encoded_count(self) -> int:
        return sum(len(batch) for batch in self.calls)


def test_repeated_texts_are_encoded_once() -> None:
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner)

    first = cache.encode(["alpha", "beta"])
    second = cache.encode(["alpha", "beta"])

    assert inner.encoded_count == 2
    np.testing.assert_array_equal(first, second)


def test_only_the_misses_are_sent_to_the_model() -> None:
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner)

    cache.encode(["alpha", "beta"])
    cache.encode(["beta", "gamma", "alpha"])

    assert inner.calls == [["alpha", "beta"], ["gamma"]]


def test_misses_are_encoded_as_one_batch_not_one_call_each() -> None:
    # Batching is most of a transformer's CPU throughput; encoding text by
    # text would throw that away.
    inner = CountingEmbedder()
    CachedEmbedder(inner).encode(["a", "b", "c", "d"])
    assert len(inner.calls) == 1


def test_results_keep_the_requested_order_when_hits_and_misses_mix() -> None:
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner)
    cache.encode(["beta"])  # warm one entry

    mixed = cache.encode(["alpha", "beta", "gamma"])
    expected = CountingEmbedder().encode(["alpha", "beta", "gamma"])

    np.testing.assert_array_equal(mixed, expected)


def test_hit_and_miss_counters_are_reported() -> None:
    cache = CachedEmbedder(CountingEmbedder())
    cache.encode(["alpha", "beta"])
    cache.encode(["alpha"])

    stats = cache.stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 1
    assert stats["entries"] == 2


def test_the_least_recently_used_entry_is_evicted_first() -> None:
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner, capacity=2)

    cache.encode(["a"])
    cache.encode(["b"])
    cache.encode(["a"])  # touch "a", making "b" the oldest
    cache.encode(["c"])  # evicts "b"

    inner.calls.clear()
    cache.encode(["a", "b"])
    assert inner.calls == [["b"]]


def test_capacity_is_enforced() -> None:
    cache = CachedEmbedder(CountingEmbedder(), capacity=3)
    cache.encode([f"text-{n}" for n in range(10)])
    assert cache.stats()["entries"] == 3


def test_encoding_nothing_returns_an_empty_array_of_the_right_width() -> None:
    cache = CachedEmbedder(CountingEmbedder())
    result = cache.encode([])
    assert result.shape == (0, 3)


def test_dimension_and_model_id_pass_through() -> None:
    cache = CachedEmbedder(CountingEmbedder())
    assert cache.dimension == 3
    assert cache.model_id == "counting"


def test_clearing_drops_every_entry() -> None:
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner)
    cache.encode(["alpha"])
    cache.clear()
    cache.encode(["alpha"])
    assert inner.encoded_count == 2


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        CachedEmbedder(CountingEmbedder(), capacity=0)
