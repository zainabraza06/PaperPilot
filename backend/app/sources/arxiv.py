"""arXiv connector.

arXiv exposes a single query endpoint that answers with an Atom 1.0 feed.
Quirks handled here:

* Errors are reported as a *successful* 200 response containing one entry
  whose ``id`` points at ``arxiv.org/api/errors`` — status codes alone are
  not enough to detect failure.
* Every record has an abstract (``summary``) but most have no DOI, so
  deduplication against Crossref/PubMed has to fall back to titles.
* The terms of use ask for at most one request every three seconds.
* Ids are versioned (``2101.00001v3``); the bare id is the stable one.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from app.core.text import collapse_whitespace
from app.core.xmlsafe import Element, parse_xml, text_of
from app.models.paper import Author, Paper, SourceName, make_paper_id
from app.models.search import SourceQuery
from app.sources.base import BaseHttpSource

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

_ERROR_ID_MARKER = "arxiv.org/api/errors"
_VERSION_SUFFIX = re.compile(r"v\d+$")
_ABS_URL = re.compile(r"arxiv\.org/abs/(?P<id>.+)$")


class ArxivSource(BaseHttpSource):
    """Search arXiv's public API."""

    name = SourceName.ARXIV
    display_name = "arXiv"

    def _rate_limit_interval(self) -> float:
        # arXiv's terms of use ask for one request every three seconds.
        return self.settings.arxiv_min_interval_seconds

    async def search(self, query: SourceQuery) -> list[Paper]:
        papers = await self._run_query(self._build_search_query(query.terms), query.limit)
        if not papers:
            # Same relaxation as PubMed: ANDing six terms across a preprint
            # server matches nothing long before the topic runs out of
            # relevant papers.
            relaxed = self._build_search_query(query.terms, operator="OR")
            if relaxed:
                self._log.info("no matches for ANDed terms; relaxing to OR")
                papers = await self._run_query(relaxed, query.limit)
        return papers

    async def _run_query(self, search_query: str, limit: int) -> list[Paper]:
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = await self._get(self.settings.arxiv_base_url, params=params)
        return self.parse_feed(response.text)

    async def fetch_by_doi(self, doi: str) -> Paper | None:
        """arXiv has no DOI index; the closest equivalent is a full-text hunt."""
        response = await self._get(
            self.settings.arxiv_base_url,
            params={"search_query": f'all:"{doi}"', "max_results": 5},
        )
        for paper in self.parse_feed(response.text):
            if paper.doi == doi:
                return paper
        return None

    async def fetch_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        """Direct id lookup, used when the user pastes an arXiv identifier."""
        response = await self._get(
            self.settings.arxiv_base_url,
            params={"id_list": arxiv_id, "max_results": 1},
        )
        papers = self.parse_feed(response.text)
        return papers[0] if papers else None

    # --- parsing ---------------------------------------------------------

    def parse_feed(self, payload: str) -> list[Paper]:
        """Turn an Atom feed into ``Paper`` records.

        Public and pure so the parsing logic can be tested against recorded
        fixtures without any network access.
        """
        try:
            root = parse_xml(payload)
        except Exception as exc:
            raise self._parse_error(f"could not parse Atom feed: {exc}") from exc

        papers: list[Paper] = []
        for entry in root.findall(f"{_ATOM}entry"):
            paper = self._parse_entry(entry)
            if paper is not None:
                papers.append(paper)
        return papers

    def _parse_entry(self, entry: Element) -> Paper | None:
        raw_id = text_of(entry.find(f"{_ATOM}id")) or ""
        if _ERROR_ID_MARKER in raw_id:
            # arXiv signals malformed queries with a 200 + error entry.
            message = text_of(entry.find(f"{_ATOM}summary")) or "unknown arXiv error"
            raise self._parse_error(f"arXiv rejected the query: {message}")

        title = text_of(entry.find(f"{_ATOM}title"))
        if not title:
            return None  # An entry with no title is unusable downstream.

        arxiv_id = self._extract_id(raw_id)
        doi = text_of(entry.find(f"{_ARXIV}doi"))

        authors: list[Author] = []
        for author_el in entry.findall(f"{_ATOM}author"):
            name = text_of(author_el.find(f"{_ATOM}name"))
            if not name:
                continue
            affiliation = text_of(author_el.find(f"{_ARXIV}affiliation"))
            author = Author.from_display_name(name, affiliation)
            if author:
                authors.append(author)

        categories = [
            term
            for el in entry.findall(f"{_ATOM}category")
            if (term := el.get("term"))
        ]

        return Paper(
            id=make_paper_id(self.name, arxiv_id, doi),
            doi=doi,
            source=self.name,
            source_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=collapse_whitespace(text_of(entry.find(f"{_ATOM}summary"))),
            published_date=_parse_timestamp(text_of(entry.find(f"{_ATOM}published"))),
            url=raw_id or f"https://arxiv.org/abs/{arxiv_id}",
            journal=text_of(entry.find(f"{_ARXIV}journal_ref")),
            publisher="arXiv",
            publication_type="preprint",
            keywords=categories,
            pdf_url=self._find_pdf_link(entry) or f"https://arxiv.org/pdf/{arxiv_id}",
        )

    @staticmethod
    def _extract_id(raw_id: str) -> str:
        """``http://arxiv.org/abs/2101.00001v2`` → ``2101.00001``."""
        match = _ABS_URL.search(raw_id)
        identifier = match.group("id") if match else raw_id
        return _VERSION_SUFFIX.sub("", identifier)

    @staticmethod
    def _find_pdf_link(entry: Element) -> str | None:
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" and link.get("href"):
                return link.get("href")
        return None

    @staticmethod
    def _build_search_query(terms: str, operator: str = "AND") -> str:
        """Build an arXiv ``search_query`` expression.

        Tokens are combined across all fields rather than sent as one quoted
        phrase: a phrase match on a multi-word topic returns almost nothing,
        while combined terms behave like the topic search users expect.
        """
        tokens = [token for token in re.split(r"\s+", terms.strip()) if token]
        if not tokens:
            return "all:*"
        # Quotes and backslashes would break out of the quoted field term.
        cleaned = [token.replace('"', "").replace("\\", "") for token in tokens[:12]]
        return f" {operator} ".join(f'all:"{token}"' for token in cleaned if token)


def _parse_timestamp(value: str | None) -> date | None:
    """Parse arXiv's ISO-8601 timestamps, tolerating the ``Z`` suffix."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None
