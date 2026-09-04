"""Small, dependency-free text helpers shared by the source connectors.

These live in ``core`` rather than in a connector because normalization
rules (what counts as "the same DOI", "the same title") must be identical
across sources or deduplication silently stops working.
"""

from __future__ import annotations

import re
import unicodedata

_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)

_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9<>\[\]+]+", re.IGNORECASE)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WS_PATTERN = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def collapse_whitespace(value: str | None) -> str | None:
    """Collapse all whitespace runs to single spaces; empty becomes ``None``."""
    if value is None:
        return None
    collapsed = _WS_PATTERN.sub(" ", value).strip()
    return collapsed or None


def strip_markup(value: str | None) -> str | None:
    """Remove XML/HTML/JATS tags and decode the handful of entities that matter.

    Crossref returns abstracts as JATS fragments and PubMed abstracts can
    carry inline markup; neither is useful downstream.
    """
    if value is None:
        return None
    text = _TAG_PATTERN.sub(" ", value)
    for entity, char in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&apos;", "'"),
        ("&#x2019;", "’"),  # noqa: RUF001 - the entity really is a curly quote
        ("&nbsp;", " "),
    ):
        text = text.replace(entity, char)
    return collapse_whitespace(text)


def normalize_doi(value: str | None) -> str | None:
    """Reduce any DOI spelling to its bare, lowercased form.

    Accepts ``https://doi.org/10.1/x``, ``doi:10.1/X``, ``10.1/x`` and text
    that merely contains a DOI. Returns ``None`` when nothing DOI-shaped is
    present, so callers can treat "no DOI" uniformly.
    """
    if not value:
        return None
    candidate = value.strip()
    lowered = candidate.lower()
    for prefix in _DOI_URL_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    match = _DOI_PATTERN.search(candidate)
    if not match:
        return None
    # Trailing punctuation is common when DOIs are scraped from prose.
    return match.group(0).rstrip(".,;)").lower()


def normalize_title(value: str | None) -> str:
    """Fold a title to a comparison key: lowercase, unaccented, alphanumeric.

    Used as the fallback dedupe key for records with no DOI, where the same
    work may differ only by punctuation, casing, or a trailing period.
    """
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = ascii_only.lower()
    alnum = _NON_ALNUM.sub(" ", lowered)
    return _WS_PATTERN.sub(" ", alnum).strip()


def truncate(value: str, limit: int, suffix: str = "…") -> str:
    """Truncate on a word boundary, appending an ellipsis when shortened."""
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0]
    return cut + suffix
