"""PubMed connector (NCBI E-utilities).

PubMed is a two-call API: ``esearch`` returns a list of PMIDs for a query,
and ``efetch`` returns the records for those PMIDs as MEDLINE XML. Both
calls are made here so callers see a single ``search()``.

Quirks handled here:

* Titles and abstracts contain inline markup (``<i>``, ``<sup>``,
  ``<AbstractText Label="METHODS">``), so text has to be gathered across
  child nodes rather than read off ``.text``.
* Structured abstracts arrive as several labelled ``AbstractText`` nodes
  and are re-assembled into one readable block.
* Dates are ragged: ``PubDate`` may hold Year/Month/Day, a month name, a
  season, or a free-text ``MedlineDate`` such as ``"2021 Jan-Feb"``.
* DOIs live in two different places depending on the record's vintage.
* The set can include ``PubmedBookArticle`` entries with a different shape.
* NCBI allows 3 requests/second anonymously, 10 with an API key.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from app.core.text import collapse_whitespace
from app.core.xmlsafe import Element, parse_xml, text_of
from app.models.paper import Author, Paper, SourceName, make_paper_id
from app.models.search import SourceQuery
from app.sources.base import BaseHttpSource

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_SEASONS = {"spring": 4, "summer": 7, "fall": 10, "autumn": 10, "winter": 1}
_YEAR_IN_TEXT = re.compile(r"(1[89]\d{2}|20\d{2})")


class PubMedSource(BaseHttpSource):
    """Search PubMed through the E-utilities endpoints."""

    name = SourceName.PUBMED
    display_name = "PubMed"

    def _rate_limit_interval(self) -> float:
        # NCBI: 3 requests/second anonymously, 10/second with an API key.
        return 1 / 9 if self.settings.pubmed_api_key else 1 / 2.5

    async def search(self, query: SourceQuery) -> list[Paper]:
        pmids = await self._esearch(query.terms, query.limit)
        if not pmids:
            return []
        return await self._efetch(pmids)

    async def fetch_by_doi(self, doi: str) -> Paper | None:
        """PubMed indexes DOIs under the ``[doi]`` search field."""
        pmids = await self._esearch(f"{doi}[doi]", limit=1)
        if not pmids:
            return None
        papers = await self._efetch(pmids)
        return papers[0] if papers else None

    async def fetch_by_pmid(self, pmid: str) -> Paper | None:
        """Direct PMID lookup, used when the user pastes an identifier."""
        papers = await self._efetch([pmid])
        return papers[0] if papers else None

    # --- E-utilities calls -----------------------------------------------

    async def _esearch(self, term: str, limit: int) -> list[str]:
        params = self._common_params()
        params.update(
            {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retmax": limit,
                "retstart": 0,
                "sort": "relevance",
            }
        )
        response = await self._get(f"{self.settings.pubmed_base_url}/esearch.fcgi", params=params)
        return self.parse_esearch(response.text)

    async def _efetch(self, pmids: list[str]) -> list[Paper]:
        params = self._common_params()
        params.update(
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
            }
        )
        response = await self._get(f"{self.settings.pubmed_base_url}/efetch.fcgi", params=params)
        papers = self.parse_articles(response.text)
        # efetch does not preserve the relevance ordering of esearch, so we
        # restore it here; the ranking stage still re-scores everything, but
        # a sensible order matters when the ranker is bypassed.
        order = {pmid: index for index, pmid in enumerate(pmids)}
        papers.sort(key=lambda paper: order.get(paper.source_id, len(order)))
        return papers

    def _common_params(self) -> dict[str, Any]:
        """Identification parameters NCBI asks every client to send."""
        params: dict[str, Any] = {"tool": "PaperPilot"}
        if self.settings.pubmed_api_key:
            params["api_key"] = self.settings.pubmed_api_key
        if self.settings.pubmed_email:
            params["email"] = self.settings.pubmed_email
        return params

    # --- parsing ----------------------------------------------------------

    def parse_esearch(self, payload: str) -> list[str]:
        """Extract the PMID list from an ``esearch`` JSON response."""
        try:
            body = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise self._parse_error(f"invalid esearch JSON: {exc}") from exc
        result = body.get("esearchresult")
        if not isinstance(result, dict):
            raise self._parse_error("esearchresult missing from response")
        if result.get("ERROR"):
            raise self._parse_error(f"esearch reported: {result['ERROR']}")
        idlist = result.get("idlist") or []
        return [str(pmid) for pmid in idlist if str(pmid).strip()]

    def parse_articles(self, payload: str) -> list[Paper]:
        """Turn an ``efetch`` MEDLINE XML document into ``Paper`` records.

        Public and pure so parsing can be tested against recorded fixtures
        without any network access.
        """
        try:
            root = parse_xml(payload)
        except Exception as exc:
            raise self._parse_error(f"could not parse efetch XML: {exc}") from exc

        papers: list[Paper] = []
        for article in root.findall("PubmedArticle"):
            paper = self._parse_article(article)
            if paper is not None:
                papers.append(paper)
        return papers

    def _parse_article(self, node: Element) -> Paper | None:
        citation = node.find("MedlineCitation")
        if citation is None:
            return None

        pmid = text_of(citation.find("PMID"))
        article = citation.find("Article")
        if not pmid or article is None:
            return None

        title = text_of(article.find("ArticleTitle"))
        if not title:
            return None
        title = title.rstrip(".") if title.endswith("].") else title

        doi = self._extract_doi(node, article)
        journal = article.find("Journal")

        return Paper(
            id=make_paper_id(self.name, pmid, doi),
            doi=doi,
            source=self.name,
            source_id=pmid,
            title=title,
            authors=self._parse_authors(article),
            abstract=self._parse_abstract(article),
            published_date=self._parse_date(article, journal),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            journal=self._parse_journal_title(journal),
            volume=text_of(journal.find("JournalIssue/Volume")) if journal is not None else None,
            issue=text_of(journal.find("JournalIssue/Issue")) if journal is not None else None,
            pages=text_of(article.find("Pagination/MedlinePgn"))
            or text_of(article.find("Pagination/StartPage")),
            publication_type=text_of(article.find("PublicationTypeList/PublicationType")),
            keywords=self._parse_keywords(citation),
        )

    @staticmethod
    def _extract_doi(node: Element, article: Element) -> str | None:
        """Find a DOI in either of the two places PubMed records keep one."""
        for elocation in article.findall("ELocationID"):
            if elocation.get("EIdType") == "doi":
                value = text_of(elocation)
                if value:
                    return value
        for article_id in node.findall("PubmedData/ArticleIdList/ArticleId"):
            if article_id.get("IdType") == "doi":
                value = text_of(article_id)
                if value:
                    return value
        return None

    @staticmethod
    def _parse_authors(article: Element) -> list[Author]:
        authors: list[Author] = []
        for node in article.findall("AuthorList/Author"):
            affiliation = text_of(node.find("AffiliationInfo/Affiliation"))
            collective = text_of(node.find("CollectiveName"))
            if collective:
                # Group authorship ("The SPRINT Research Group").
                author = Author.from_display_name(collective, affiliation)
            else:
                author = Author.from_parts(
                    text_of(node.find("ForeName")) or text_of(node.find("Initials")),
                    text_of(node.find("LastName")),
                    affiliation,
                )
            if author:
                authors.append(author)
        return authors

    @staticmethod
    def _parse_abstract(article: Element) -> str | None:
        """Reassemble a possibly-structured abstract into one block of text.

        Labelled sections are kept as ``LABEL: text`` because the labels
        carry real information (Methods vs. Conclusions) that both the
        ranker and the summarizer benefit from.
        """
        sections: list[str] = []
        for node in article.findall("Abstract/AbstractText"):
            text = text_of(node)
            if not text:
                continue
            label = node.get("Label")
            sections.append(f"{label.strip().title()}: {text}" if label else text)
        if not sections:
            # Some records carry only a publisher-supplied "other" abstract.
            other = text_of(article.find("OtherAbstract/AbstractText"))
            return collapse_whitespace(other)
        return collapse_whitespace(" ".join(sections))

    @staticmethod
    def _parse_journal_title(journal: Element | None) -> str | None:
        if journal is None:
            return None
        return text_of(journal.find("Title")) or text_of(journal.find("ISOAbbreviation"))

    @staticmethod
    def _parse_keywords(citation: Element) -> list[str]:
        """Prefer author keywords, and fall back to MeSH descriptors."""
        keywords = [
            text for node in citation.findall("KeywordList/Keyword") if (text := text_of(node))
        ]
        if keywords:
            return keywords
        return [
            text
            for node in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
            if (text := text_of(node))
        ]

    @classmethod
    def _parse_date(cls, article: Element, journal: Element | None) -> date | None:
        """Resolve a publication date from PubMed's several date encodings.

        ``ArticleDate`` (electronic publication) is preferred because it is
        always a complete Y/M/D; ``PubDate`` is the fallback and may be
        year-only, season-based, or free text.
        """
        electronic = article.find("ArticleDate")
        if electronic is not None:
            parsed = cls._date_from_parts(
                text_of(electronic.find("Year")),
                text_of(electronic.find("Month")),
                text_of(electronic.find("Day")),
            )
            if parsed:
                return parsed

        pub_date = journal.find("JournalIssue/PubDate") if journal is not None else None
        if pub_date is None:
            return None

        parsed = cls._date_from_parts(
            text_of(pub_date.find("Year")),
            text_of(pub_date.find("Month")),
            text_of(pub_date.find("Day")),
        )
        if parsed:
            return parsed

        # MedlineDate holds free text such as "2021 Jan-Feb" or "1998-1999".
        medline = text_of(pub_date.find("MedlineDate"))
        if medline:
            year_match = _YEAR_IN_TEXT.search(medline)
            if year_match:
                month_match = re.search(r"[A-Za-z]{3,}", medline)
                return cls._date_from_parts(
                    year_match.group(1),
                    month_match.group(0) if month_match else None,
                    None,
                )
        return None

    @staticmethod
    def _date_from_parts(year: str | None, month: str | None, day: str | None) -> date | None:
        if not year or not year.strip().isdigit():
            return None
        month_number = 1
        if month:
            token = month.strip().lower()
            if token.isdigit():
                month_number = int(token)
            else:
                month_number = _MONTHS.get(token[:3], _SEASONS.get(token, 1))
        day_number = int(day) if day and day.strip().isdigit() else 1
        try:
            return date(int(year), max(1, min(month_number, 12)), max(1, min(day_number, 28)))
        except ValueError:
            return None
