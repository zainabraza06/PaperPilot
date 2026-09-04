"""Tests for the shared normalization helpers.

These rules have to be identical across connectors or deduplication
silently stops working, so they are pinned here rather than per-source.
"""

from __future__ import annotations

import pytest

from app.core.text import (
    collapse_whitespace,
    normalize_doi,
    normalize_title,
    strip_markup,
    truncate,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1038/s41587-022-01234-5", "10.1038/s41587-022-01234-5"),
        ("10.1038/S41587-022-01234-5", "10.1038/s41587-022-01234-5"),
        ("https://doi.org/10.1038/s41587-022-01234-5", "10.1038/s41587-022-01234-5"),
        ("http://dx.doi.org/10.1038/s41587-022-01234-5", "10.1038/s41587-022-01234-5"),
        ("doi:10.1038/s41587-022-01234-5", "10.1038/s41587-022-01234-5"),
        ("  10.1038/s41587-022-01234-5.  ", "10.1038/s41587-022-01234-5"),
        ("see 10.1234/abc for details", "10.1234/abc"),
        ("not-a-doi", None),
        ("", None),
        (None, None),
    ],
)
def test_doi_normalization(raw: str | None, expected: str | None) -> None:
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Attention Is All You Need", "attention is all you need."),
        ("Prime editing: a review", "Prime editing - a review"),
        ("Rôle of CRISPR", "Role of CRISPR"),
        ("Multi   spaced\ntitle", "Multi spaced title"),
    ],
)
def test_titles_that_should_compare_equal(left: str, right: str) -> None:
    assert normalize_title(left) == normalize_title(right)


def test_titles_that_should_not_compare_equal() -> None:
    assert normalize_title("Graph neural networks") != normalize_title("Graph neural network")


def test_strip_markup_removes_jats_and_html_tags() -> None:
    assert strip_markup("<jats:p>Hello <jats:italic>world</jats:italic></jats:p>") == "Hello world"


def test_strip_markup_decodes_common_entities() -> None:
    assert strip_markup("Smith &amp; Jones &lt;2020&gt;") == "Smith & Jones <2020>"


def test_collapse_whitespace_returns_none_for_blank_input() -> None:
    assert collapse_whitespace("   \n ") is None
    assert collapse_whitespace(None) is None


def test_truncate_breaks_on_a_word_boundary() -> None:
    assert truncate("the quick brown fox", 12) == "the quick…"
    assert truncate("short", 12) == "short"
