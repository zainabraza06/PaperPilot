"""Query interpretation.

The product accepts four kinds of input in one box — a topic, a keyword, an
identifier (DOI / arXiv id / PMID), or a pasted abstract snippet — so the
first thing the pipeline does is work out which one it just received.

This matters for two reasons:

1. An identifier should trigger a direct lookup, not a relevance search.
2. No upstream API accepts a 200-word paragraph as a query, so a pasted
   abstract has to be distilled into keywords before fan-out. The full text
   is preserved for Stage 2, which embeds it for semantic ranking.
"""

from __future__ import annotations

import re

from app.core.text import normalize_doi
from app.models.search import IdentifierKind, ParsedQuery, QueryIntent

# ``2101.00001`` / ``2101.00001v3`` (post-2007) and ``math.GT/0309136`` (pre-2007).
_ARXIV_NEW = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
_ARXIV_OLD = re.compile(r"^(?:arxiv:)?([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.IGNORECASE)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", re.IGNORECASE)
_PMID = re.compile(r"^(?:pmid:?\s*)?(\d{1,8})$", re.IGNORECASE)
_PUBMED_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)

# A pasted abstract is distinguished from a topic by length alone; this is
# the point where "a long topic" stops being plausible.
_SNIPPET_WORD_THRESHOLD = 25
_KEYWORD_WORD_THRESHOLD = 2

# Deliberately small: a general-purpose English stoplist plus the filler
# verbs that dominate abstract prose ("we show that...", "results suggest").
_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can cannot could did do does doing down during each few for
    from further had has have having he her here hers herein him his how however i if in into is
    it its itself just may me might more most must my no nor not of off on once only or other
    ought our ours out over own same shall she should so some such than that the their theirs
    them then there these they this those through to too under until up upon us very was we were
    what when where whether which while who whom why will with within would you your yours
    study studies result results show shows shown showed suggest suggests using used use based
    approach approaches paper papers method methods finding findings conclusion conclusions
    background objective objectives purpose data analysis significant significantly compared
    """.split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")


def parse_query(raw: str) -> ParsedQuery:
    """Classify a user's raw input and derive the terms to send upstream."""
    text = raw.strip()
    if not text:
        raise ValueError("query must not be empty")

    identifier = _detect_identifier(text)
    if identifier is not None:
        kind, value = identifier
        return ParsedQuery(
            raw=text,
            intent=QueryIntent.IDENTIFIER,
            search_terms=value,
            identifier_kind=kind,
            identifier=value,
            keywords=[value],
        )

    words = text.split()
    if len(words) > _SNIPPET_WORD_THRESHOLD:
        keywords = extract_keywords(text)
        return ParsedQuery(
            raw=text,
            intent=QueryIntent.ABSTRACT_SNIPPET,
            # Upstream APIs get a distilled keyword query; the full snippet
            # is still carried in ``raw`` for semantic ranking.
            search_terms=" ".join(keywords),
            keywords=keywords,
        )

    intent = QueryIntent.KEYWORD if len(words) <= _KEYWORD_WORD_THRESHOLD else QueryIntent.TOPIC
    return ParsedQuery(
        raw=text,
        intent=intent,
        search_terms=text,
        keywords=extract_keywords(text) or [w.lower() for w in words],
    )


def extract_keywords(text: str, limit: int = 10) -> list[str]:
    """Pull the most distinctive terms out of a block of text.

    Frequency-ranked, stopword-filtered, order-stable. This is intentionally
    a simple lexical heuristic rather than a model: it only has to produce a
    query the three APIs can answer, and the semantic ranking in Stage 2 is
    what actually decides relevance.
    """
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for position, match in enumerate(_WORD.finditer(text)):
        token = match.group(0).lower().strip("-")
        if len(token) < 3 or token in _STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
        first_seen.setdefault(token, position)

    ranked = sorted(counts, key=lambda term: (-counts[term], first_seen[term]))
    return ranked[:limit]


def _detect_identifier(text: str) -> tuple[IdentifierKind, str] | None:
    """Return the identifier this query *is*, if any.

    Only whole-input matches count. A DOI mentioned inside a paragraph is
    not a lookup request — the paragraph is the query.
    """
    doi = normalize_doi(text)
    if doi and len(text.split()) == 1:
        return IdentifierKind.DOI, doi

    url_match = _ARXIV_URL.search(text)
    if url_match:
        return IdentifierKind.ARXIV, _VERSION_SUFFIX.sub("", url_match.group(1).removesuffix(".pdf"))

    pubmed_match = _PUBMED_URL.search(text)
    if pubmed_match:
        return IdentifierKind.PMID, pubmed_match.group(1)

    collapsed = text.replace(" ", "") if text.lower().startswith(("arxiv:", "pmid")) else text
    for pattern in (_ARXIV_NEW, _ARXIV_OLD):
        match = pattern.match(collapsed)
        if match:
            return IdentifierKind.ARXIV, match.group(1)

    pmid_match = _PMID.match(collapsed)
    if pmid_match:
        return IdentifierKind.PMID, pmid_match.group(1)

    return None
