"""RIS export.

RIS is a line-oriented format with a rigid grammar that is easy to get
subtly wrong, and reference managers are unforgiving about it:

* A tag line is **exactly** two uppercase characters, two spaces, a
  hyphen, and a space — ``TY  - JOUR``. One space instead of two and
  EndNote silently drops the field.
* ``TY`` must be the first line of a record and ``ER  -`` the last.
* The specification calls for CRLF line endings.
* Repeatable tags (``AU``, ``KW``) are emitted once per value rather than
  joined.

All four are asserted by tests, and the output is round-tripped through an
independent RIS parser so "correct" means "another implementation agrees",
not "it looks right".
"""

from __future__ import annotations

from app.models.paper import Paper

#: The specification mandates CRLF.
_CRLF = "\r\n"

#: Separators seen in deposited page ranges. The third is U+2013 EN DASH,
#: written as ``chr()`` rather than inline because an en-dash and a hyphen
#: are indistinguishable in most editors — and publishers really do deposit
#: ranges with a typographic dash, which an ASCII-only split would emit as
#: a single start page.
_PAGE_SEPARATORS = ("--", "-", chr(0x2013))

#: Publication types mapped onto RIS reference types.
_REFERENCE_TYPES = {
    "journal-article": "JOUR",
    "journal article": "JOUR",
    "proceedings-article": "CONF",
    "conference paper": "CONF",
    "book-chapter": "CHAP",
    "book": "BOOK",
    "monograph": "BOOK",
    "edited-book": "BOOK",
    "reference-entry": "CHAP",
    "dissertation": "THES",
    "report": "RPRT",
    "posted-content": "UNPB",
    "preprint": "UNPB",
}


class RISFormatter:
    """Renders papers as an RIS file."""

    format_id = "ris"
    extension = "ris"
    media_type = "application/x-research-info-systems"

    def format_many(self, papers: list[Paper]) -> str:
        return "".join(self._record(paper) for paper in papers)

    def format_one(self, paper: Paper) -> str:
        return self._record(paper)

    def _record(self, paper: Paper) -> str:
        lines: list[tuple[str, str]] = [("TY", self._reference_type(paper))]

        for author in paper.authors:
            # RIS wants "Family, Given"; a name we could not split is
            # emitted whole rather than guessed at.
            if author.family and author.given:
                lines.append(("AU", f"{author.family}, {author.given}"))
            else:
                lines.append(("AU", author.name))

        lines.append(("TI", _clean(paper.title)))

        if paper.journal:
            # T2 is the secondary title (container); JO is the full journal
            # name. Managers differ on which they read, so both are given.
            lines.append(("T2", _clean(paper.journal)))
            lines.append(("JO", _clean(paper.journal)))
        if paper.abstract:
            lines.append(("AB", _clean(paper.abstract)))
        if paper.published_date:
            lines.append(("PY", str(paper.published_date.year)))
            # DA is a full date in YYYY/MM/DD/ form; the trailing slash is
            # part of the format, not a typo.
            lines.append(("DA", paper.published_date.strftime("%Y/%m/%d/")))
        if paper.volume:
            lines.append(("VL", paper.volume))
        if paper.issue:
            lines.append(("IS", paper.issue))

        start, end = _split_pages(paper.pages)
        if start:
            lines.append(("SP", start))
        if end:
            lines.append(("EP", end))

        if paper.publisher:
            lines.append(("PB", _clean(paper.publisher)))
        if paper.doi:
            lines.append(("DO", paper.doi))
        if paper.url:
            lines.append(("UR", paper.url))
        for keyword in paper.keywords:
            lines.append(("KW", _clean(keyword)))
        lines.append(("DB", "PaperPilot"))
        lines.append(("DP", "+".join(source.value for source in paper.all_sources)))

        body = _CRLF.join(f"{tag}  - {value}" for tag, value in lines)
        # ER carries no value but still takes the tag form, and the record
        # is terminated by a blank line.
        return body + _CRLF + "ER  - " + _CRLF + _CRLF

    @staticmethod
    def _reference_type(paper: Paper) -> str:
        raw = (paper.publication_type or "").strip().lower()
        if raw in _REFERENCE_TYPES:
            return _REFERENCE_TYPES[raw]
        if paper.source.value == "arxiv" and not paper.journal:
            return "UNPB"
        return "JOUR" if paper.journal else "GEN"


def _clean(value: str) -> str:
    """Collapse newlines: a line break would be read as a new tag line."""
    return " ".join(value.split())


def _split_pages(pages: str | None) -> tuple[str | None, str | None]:
    """Split ``1021-1030`` into start and end pages.

    A single page number is a start page with no end, and anything that
    does not look like a range is passed through as the start page rather
    than dropped.
    """
    if not pages:
        return None, None
    text = pages.strip()
    for separator in _PAGE_SEPARATORS:
        if separator in text:
            start, _, end = text.partition(separator)
            return start.strip() or None, end.strip() or None
    return text or None, None
