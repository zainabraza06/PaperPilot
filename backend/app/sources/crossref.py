"""Crossref connector.

Crossref indexes DOI-registered works across essentially every publisher,
which makes it the best source for *identifier* lookups and the weakest for
abstracts — publishers are not required to deposit them, so a large share of
records come back with no abstract at all.

Quirks handled here:

* Abstracts are deposited as JATS XML fragments and must be de-tagged.
* ``issued.date-parts`` is a ragged array: ``[[2021]]``, ``[[2021, 5]]``,
  ``[[2021, 5, 3]]`` and ``[[None]]`` are all real responses.
* Titles and container titles are arrays that are occasionally empty.
* Supplying a ``mailto`` moves us into the faster "polite" request pool.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from app.core.errors import SourceUnavailableError
from app.core.text import collapse_whitespace, strip_markup
from app.models.paper import Author, Paper, SourceName, make_paper_id
from app.models.search import SourceQuery
from app.sources.base import BaseHttpSource

# Crossref indexes far more than papers, and the non-paper records are not
# harmless noise: a `peer-review` record's title *quotes the reviewed paper's
# title*, so "Decision letter: Escape from neutralizing antibodies by SARS-CoV-2
# spike..." scores at or near the top on every relevance signal there is. On one
# evaluation query, all eight Crossref results were review reports. `component`
# records are figures and supplementary files; the container types are journals
# and volumes rather than anything inside them.
#
# The allow-list is applied server-side, so the row budget is spent on usable
# records instead of being silently eaten by junk. Filtering at the source
# rather than in the ranker is deliberate: a ranker cannot fix bad candidates,
# it can only reorder them.
_ALLOWED_TYPES = (
    "journal-article",
    "posted-content",  # preprints
    "proceedings-article",
    "book-chapter",
    "monograph",
    "reference-entry",
    "report",
    "dissertation",
)

#: Repeated same-field filters are OR-ed by the Crossref API.
_TYPE_FILTER = ",".join(f"type:{name}" for name in _ALLOWED_TYPES)

#: Defence in depth for paths with no server-side filter, such as DOI lookup.
_EXCLUDED_TYPES = frozenset(
    {
        "peer-review",
        "component",
        "grant",
        "dataset",
        "standard",
        "journal",
        "journal-issue",
        "journal-volume",
        "book-series",
        "book-set",
        "proceedings",
        "proceedings-series",
        "report-series",
    }
)

# Trimming the payload to the fields we model keeps responses far smaller.
_SELECT_FIELDS = ",".join(
    [
        "DOI",
        "title",
        "author",
        "abstract",
        "issued",
        "published-print",
        "published-online",
        "container-title",
        "URL",
        "type",
        "publisher",
        "volume",
        "issue",
        "page",
        "subject",
        "is-referenced-by-count",
        "link",
    ]
)


class CrossrefSource(BaseHttpSource):
    """Search the Crossref REST API."""

    name = SourceName.CROSSREF
    display_name = "Crossref"

    # Crossref's public pool tolerates roughly 50 req/s; we stay well under.
    min_request_interval = 0.05

    async def search(self, query: SourceQuery) -> list[Paper]:
        params: dict[str, Any] = {
            "query.bibliographic": query.terms,
            "rows": query.limit,
            "filter": _TYPE_FILTER,
            "select": _SELECT_FIELDS,
            "sort": "relevance",
            "order": "desc",
        }
        if self.settings.crossref_mailto:
            params["mailto"] = self.settings.crossref_mailto

        response = await self._get(f"{self.settings.crossref_base_url}/works", params=params)
        return self.parse_work_list(response.text)

    async def fetch_by_doi(self, doi: str) -> Paper | None:
        params: dict[str, Any] = {}
        if self.settings.crossref_mailto:
            params["mailto"] = self.settings.crossref_mailto
        try:
            response = await self._get(
                f"{self.settings.crossref_base_url}/works/{doi}", params=params
            )
        except SourceUnavailableError as exc:
            # A DOI Crossref does not know surfaces as a 404. That is
            # "not found", not a failure of the source.
            if "HTTP 404" in str(exc):
                return None
            raise
        return self.parse_single_work(response.text)

    # --- parsing ---------------------------------------------------------

    def parse_work_list(self, payload: str) -> list[Paper]:
        """Parse a ``/works`` response into ``Paper`` records."""
        body = self._load(payload)
        message = body.get("message")
        items = message.get("items") if isinstance(message, dict) else None
        if not isinstance(items, list):
            raise self._parse_error("message.items was missing or not a list")

        papers: list[Paper] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            paper = self.parse_work(item)
            if paper is not None:
                papers.append(paper)
        return papers

    def parse_single_work(self, payload: str) -> Paper | None:
        """Parse a ``/works/{doi}`` response."""
        body = self._load(payload)
        message = body.get("message")
        if not isinstance(message, dict):
            raise self._parse_error("message was not an object")
        return self.parse_work(message)

    def parse_work(self, item: dict[str, Any]) -> Paper | None:
        """Normalize one Crossref work. Returns ``None`` if unusably sparse."""
        if item.get("type") in _EXCLUDED_TYPES:
            return None

        title = _first_string(item.get("title"))
        doi = item.get("DOI")
        if not title or not doi:
            # Crossref also carries stub records with no title at all;
            # without one there is nothing to rank, display or cite.
            return None
        doi = str(doi)

        authors: list[Author] = []
        for entry in item.get("author") or []:
            if not isinstance(entry, dict):
                continue
            affiliation = _first_affiliation(entry.get("affiliation"))
            author = Author.from_parts(entry.get("given"), entry.get("family"), affiliation)
            if author is None and entry.get("name"):
                # Consortium and organisation authors carry a single ``name``.
                author = Author.from_display_name(str(entry["name"]), affiliation)
            if author:
                authors.append(author)

        return Paper(
            id=make_paper_id(self.name, doi, doi),
            doi=doi,
            source=self.name,
            source_id=doi,
            title=title,
            authors=authors,
            abstract=strip_markup(item.get("abstract")),
            published_date=_parse_issued(item),
            url=item.get("URL") or f"https://doi.org/{doi}",
            journal=_first_string(item.get("container-title")),
            publisher=collapse_whitespace(item.get("publisher")),
            volume=collapse_whitespace(item.get("volume")),
            issue=collapse_whitespace(item.get("issue")),
            pages=collapse_whitespace(item.get("page")),
            publication_type=item.get("type"),
            keywords=[s for s in (item.get("subject") or []) if isinstance(s, str)],
            pdf_url=_find_pdf_link(item.get("link")),
            citation_count=_as_int(item.get("is-referenced-by-count")),
        )

    def _load(self, payload: str) -> dict[str, Any]:
        try:
            body = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise self._parse_error(f"invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise self._parse_error("top-level response was not an object")
        return body


def _first_string(value: Any) -> str | None:
    """Crossref returns most single-valued text fields as arrays.

    Markup is stripped here rather than at each call site because *every*
    text field Crossref deposits can carry inline JATS — publishers send
    ``<i>``, ``<sub>``, ``<sup>`` and ``<scp>`` in titles and journal names
    as readily as in abstracts.

    Leaving it in was visibly wrong in three places at once, which is why
    it is handled once at the boundary:

    * The frontend escapes it, so a reader saw a literal
      ``Modeling <i>FGFR2</i> -Linked Craniosynostosis``.
    * It **broke deduplication.** The title fallback key folds non-alphanumerics
      to spaces, so the tags survived as the tokens ``i``/``sub``, and the
      Crossref copy of a paper stopped matching the PubMed copy of the same
      paper — the exact case that fallback exists for.
    * It reached BibTeX and RIS export, where it is not valid in either.
    """
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if not isinstance(entry, str):
            continue
        # strip_markup collapses whitespace itself, and returns None for a
        # value that was nothing but tags.
        cleaned = strip_markup(entry)
        if cleaned:
            return cleaned
    return None


def _first_affiliation(value: Any) -> str | None:
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict) and entry.get("name"):
                return collapse_whitespace(str(entry["name"]))
    return None


def _parse_issued(item: dict[str, Any]) -> date | None:
    """Resolve a publication date from Crossref's several date fields.

    ``issued`` is the canonical "when was this published" field, but it is
    sometimes absent or year-only; the print/online dates are the fallbacks.
    Partial dates are padded to the first of the month rather than dropped,
    because year-level precision is still useful for sorting and citations.
    """
    for key in ("issued", "published-print", "published-online", "created"):
        container = item.get(key)
        if not isinstance(container, dict):
            continue
        parts = container.get("date-parts")
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        numbers = [n for n in parts[0] if isinstance(n, int)]
        if not numbers:
            continue
        year = numbers[0]
        month = numbers[1] if len(numbers) > 1 else 1
        day = numbers[2] if len(numbers) > 2 else 1
        try:
            return date(year, max(1, min(month, 12)), max(1, min(day, 28)))
        except ValueError:
            continue
    return None


def _find_pdf_link(links: Any) -> str | None:
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("content-type") == "application/pdf" and link.get("URL"):
            return str(link["URL"])
    return None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
