"""Plain-text citations, in APA 7th and Vancouver.

Two styles rather than one because the audience is split: APA is the
default across most of science, Vancouver is what biomedical journals
require, and PaperPilot searches PubMed. Both have precise author rules
that are the usual source of wrong output:

* **APA 7th** lists up to 20 authors, joins the last with ``&``, and for
  21 or more gives the first 19, an ellipsis, then the *final* author —
  not the twentieth. Initials are period-separated.
* **Vancouver** lists up to 6 authors then ``et al.``, uses no
  punctuation between surname and initials, and no ampersand.

Getting these wrong is not cosmetic: a citation with the wrong author
count is a citation a journal will reject.
"""

from __future__ import annotations

from enum import Enum

from app.models.paper import Author, Paper


class TextStyle(str, Enum):
    APA = "apa"
    VANCOUVER = "vancouver"


#: APA: list every author up to this many.
_APA_MAX_LISTED = 20
#: Beyond that, list this many, then an ellipsis, then the last author.
_APA_HEAD = 19
#: Vancouver: list this many, then "et al.".
_VANCOUVER_MAX = 6


class PlainTextFormatter:
    """Renders papers as human-readable citations."""

    format_id = "text"
    extension = "txt"
    media_type = "text/plain"

    def __init__(self, style: TextStyle = TextStyle.APA) -> None:
        self._style = style

    @property
    def style(self) -> TextStyle:
        return self._style

    def format_many(self, papers: list[Paper]) -> str:
        if self._style is TextStyle.VANCOUVER:
            # Vancouver citations are numbered, and the number is part of
            # the citation rather than decoration.
            return "\n".join(
                f"{index}. {self.format_one(paper)}"
                for index, paper in enumerate(papers, start=1)
            )
        return "\n\n".join(self.format_one(paper) for paper in papers)

    def format_one(self, paper: Paper) -> str:
        if self._style is TextStyle.VANCOUVER:
            return self._vancouver(paper)
        return self._apa(paper)

    # --- APA 7th -----------------------------------------------------------

    def _apa(self, paper: Paper) -> str:
        authors = self._apa_authors(paper.authors)
        year = paper.published_date.year if paper.published_date else "n.d."
        title = paper.title.rstrip(".")

        parts = [f"{authors} ({year}). {title}."] if authors else [f"{title}. ({year})."]

        if paper.journal:
            venue = paper.journal
            if paper.volume:
                venue += f", {paper.volume}"
                if paper.issue:
                    venue += f"({paper.issue})"
            if paper.pages:
                venue += f", {paper.pages}"
            parts.append(f"{venue}.")

        if paper.doi:
            parts.append(f"https://doi.org/{paper.doi}")
        elif paper.url:
            parts.append(paper.url)
        return " ".join(parts)

    @staticmethod
    def _apa_authors(authors: list[Author]) -> str:
        if not authors:
            return ""
        formatted = [_apa_name(author) for author in authors]

        if len(formatted) == 1:
            return formatted[0]
        if len(formatted) <= _APA_MAX_LISTED:
            return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
        # 21 or more: first 19, ellipsis, then the final author. A common
        # bug is to use the 20th here instead of the last.
        return ", ".join(formatted[:_APA_HEAD]) + f", ... {formatted[-1]}"

    # --- Vancouver ---------------------------------------------------------

    def _vancouver(self, paper: Paper) -> str:
        names = [_vancouver_name(author) for author in paper.authors]
        if len(names) > _VANCOUVER_MAX:
            authors = ", ".join(names[:_VANCOUVER_MAX]) + ", et al"
        else:
            authors = ", ".join(names)

        title = paper.title.rstrip(".")
        parts = [f"{authors}. {title}." if authors else f"{title}."]

        if paper.journal:
            year = paper.published_date.year if paper.published_date else ""
            venue = f"{paper.journal}."
            if year:
                venue += f" {year}"
            if paper.volume:
                venue += f";{paper.volume}"
                if paper.issue:
                    venue += f"({paper.issue})"
            if paper.pages:
                venue += f":{paper.pages}"
            parts.append(venue + ".")
        elif paper.published_date:
            parts.append(f"{paper.published_date.year}.")

        if paper.doi:
            parts.append(f"doi:{paper.doi}")
        return " ".join(parts)


def _apa_name(author: Author) -> str:
    """``Chen, W.`` — surname, then period-separated initials."""
    family = author.family or author.name
    if not author.given:
        return family
    initials = " ".join(f"{part[0]}." for part in author.given.split() if part)
    return f"{family}, {initials}" if initials else family


def _vancouver_name(author: Author) -> str:
    """``Chen W`` — surname, space, initials, no periods."""
    family = author.family or author.name
    if not author.given:
        return family
    initials = "".join(part[0] for part in author.given.split() if part)
    return f"{family} {initials}" if initials else family
