"""BibTeX export.

BibTeX is easy to produce approximately and surprisingly fiddly to produce
correctly. The four things most implementations get wrong, all handled
here and all covered by round-trip tests against an independent parser:

1. **Escaping.** ``& % $ # _ { } ~ ^ \\`` are LaTeX control characters. An
   unescaped ``&`` in a title — common in journal names like "Cell & Gene
   Therapy" — breaks the user's build, not ours, which is the worst place
   for a bug to surface.
2. **Title case protection.** Most BibTeX styles lowercase titles. Without
   brace protection "CRISPR-Cas9 in T cells" is typeset as "Crispr-cas9 in
   t cells", quietly corrupting every acronym and gene symbol in a
   bibliography. Capitalised tokens are wrapped in braces.
3. **Citation keys.** They must be unique within a file, ASCII, and free
   of BibTeX's delimiters. Two papers by the same author in the same year
   collide constantly, so keys are disambiguated with a suffix.
4. **Entry types.** A preprint is not an ``@article``. The type is derived
   from the record's own publication type rather than defaulted.
"""

from __future__ import annotations

import re
import unicodedata

from app.models.paper import Author, Paper

#: LaTeX control characters and their replacements.
#:
#: Substituted in a single regex pass rather than by chained ``str.replace``
#: calls. Sequential replacement corrupts its own output: escaping the
#: backslash first produces ``\textbackslash{}``, and the later brace rules
#: then escape *those* braces into ``\textbackslash\{\}``. Replacing every
#: source character exactly once, with no re-scanning, is the only
#: order-independent way to get this right — and a round-trip test against
#: an independent parser is what caught it.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_PATTERN = re.compile("[" + re.escape("".join(_LATEX_ESCAPES)) + "]")

#: Crossref/PubMed publication types mapped onto BibTeX entry types.
_ENTRY_TYPES = {
    "journal-article": "article",
    "journal article": "article",
    "proceedings-article": "inproceedings",
    "conference paper": "inproceedings",
    "book-chapter": "incollection",
    "book": "book",
    "monograph": "book",
    "edited-book": "book",
    "reference-entry": "incollection",
    "dissertation": "phdthesis",
    "report": "techreport",
    "posted-content": "misc",
    "preprint": "misc",
}

_MONTH_MACROS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)

_KEY_SAFE = re.compile(r"[^A-Za-z0-9]+")
#: A token worth protecting: contains an interior capital or a digit, or is
#: all-caps. Plain sentence-initial capitalisation is left alone, since
#: protecting every first word would fight the style rather than help it.
_NEEDS_PROTECTION = re.compile(r"^[A-Za-z]*[A-Z0-9][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*[.,;:]?$")


class BibTeXFormatter:
    """Renders papers as a BibTeX database."""

    format_id = "bibtex"
    extension = "bib"
    media_type = "application/x-bibtex"

    def format_many(self, papers: list[Paper]) -> str:
        """Render several papers, guaranteeing unique citation keys."""
        used: dict[str, int] = {}
        entries = []
        for paper in papers:
            key = self._unique_key(paper, used)
            entries.append(self._entry(paper, key))
        return "\n\n".join(entries) + "\n" if entries else ""

    def format_one(self, paper: Paper) -> str:
        return self.format_many([paper])

    # --- entry construction ------------------------------------------------

    def _entry(self, paper: Paper, key: str) -> str:
        entry_type = self._entry_type(paper)
        fields: list[tuple[str, str]] = []

        def add(
            name: str,
            value: str | None,
            *,
            protect: bool = False,
            pre_escaped: bool = False,
        ) -> None:
            """Add a field, escaping unless the caller already did.

            ``pre_escaped`` exists for the author field, which escapes each
            name itself and then adds *structural* braces around
            unsplittable names. Escaping that again turns those braces into
            literal ``\\{`` and ``\\}``, so BibTeX stops seeing a grouped
            name and starts seeing punctuation.
            """
            if not value:
                return
            rendered = value if pre_escaped else escape(value)
            if protect:
                rendered = protect_case(rendered)
            fields.append((name, rendered))

        add("title", paper.title, protect=True)
        add("author", self._authors(paper.authors), pre_escaped=True)
        if entry_type == "incollection" or entry_type == "inproceedings":
            add("booktitle", paper.journal, protect=True)
        else:
            add("journal", paper.journal, protect=True)
        if paper.published_date:
            fields.append(("year", str(paper.published_date.year)))
        add("volume", paper.volume)
        add("number", paper.issue)
        add("pages", _page_range(paper.pages))
        add("publisher", paper.publisher)
        add("doi", paper.doi)
        add("url", paper.url)
        add("abstract", paper.abstract)
        if paper.keywords:
            add("keywords", ", ".join(paper.keywords))
        add("note", _note(paper))

        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
        # The month macro is unquoted on purpose: `month = jul` resolves to a
        # style-aware abbreviation, while `month = {jul}` is a literal string.
        if paper.published_date:
            body += f",\n  month = {_MONTH_MACROS[paper.published_date.month - 1]}"
        return f"@{entry_type}{{{key},\n{body}\n}}"

    @staticmethod
    def _entry_type(paper: Paper) -> str:
        raw = (paper.publication_type or "").strip().lower()
        if raw in _ENTRY_TYPES:
            return _ENTRY_TYPES[raw]
        # An arXiv record with no journal is a preprint whatever it claims.
        if paper.source.value == "arxiv" and not paper.journal:
            return "misc"
        return "article" if paper.journal else "misc"

    @staticmethod
    def _authors(authors: list[Author]) -> str:
        """``Last, First and Last, First`` — BibTeX's own name grammar.

        The ``and`` separator is structural, not punctuation: it is how
        BibTeX finds name boundaries. A comma-separated list would be read
        as a single author with a very long surname.
        """
        parts = []
        for author in authors:
            if author.family and author.given:
                parts.append(f"{escape(author.family)}, {escape(author.given)}")
            else:
                # Consortium and single-token names are brace-wrapped so
                # BibTeX does not try to split them into first/last.
                parts.append("{" + escape(author.name) + "}")
        return " and ".join(parts)

    def _unique_key(self, paper: Paper, used: dict[str, int]) -> str:
        """Build a stable, collision-free citation key.

        ``surname_word_year`` reads well in a document and is what most
        reference managers produce. Collisions are real — two papers by the
        same group in the same year is the normal case, not an edge case —
        so a repeat gets a letter suffix.
        """
        base = self._key_base(paper)
        count = used.get(base, 0)
        used[base] = count + 1
        if count == 0:
            return base
        # a, b, c ... matching the convention of every reference manager.
        return f"{base}{chr(ord('a') + count - 1)}"

    @staticmethod
    def _key_base(paper: Paper) -> str:
        surname = ""
        if paper.authors:
            first = paper.authors[0]
            surname = first.family or first.name.split()[-1]
        surname = _ascii_only(surname).lower()

        word = ""
        for token in _ascii_only(paper.title).lower().split():
            cleaned = _KEY_SAFE.sub("", token)
            if len(cleaned) > 3:
                word = cleaned
                break

        year = str(paper.published_date.year) if paper.published_date else "nodate"
        # Sanitize each part *before* joining. Doing it after would strip
        # the underscores that were just inserted as separators.
        parts = [cleaned for part in (surname, word, year) if (cleaned := _KEY_SAFE.sub("", part))]
        return "_".join(parts) or "paper"


def escape(value: str) -> str:
    """Escape LaTeX control characters so the output compiles."""
    return _LATEX_PATTERN.sub(lambda match: _LATEX_ESCAPES[match.group(0)], value)


def protect_case(value: str) -> str:
    """Brace-protect tokens whose capitalisation carries meaning.

    Applied to titles and venue names, where a BibTeX style would
    otherwise lowercase "CRISPR" to "crispr" and "T cells" to "t cells".
    Only tokens with an interior capital, a digit, or full capitalisation
    are protected — protecting ordinary words would defeat the style
    instead of cooperating with it.
    """
    protected = []
    for token in value.split(" "):
        if not token or token.startswith("{"):
            protected.append(token)
            continue
        # Trailing punctuation is left outside the braces: "{DNA}." rather
        # than "{DNA.}". The braces exist to protect capitalisation, and a
        # full stop has none to protect.
        core = token.rstrip(".,;:)")
        suffix = token[len(core) :]
        if core and _NEEDS_PROTECTION.match(core) and any(c.isupper() or c.isdigit() for c in core[1:]):
            protected.append("{" + core + "}" + suffix)
        else:
            protected.append(token)
    return " ".join(protected)


def _page_range(pages: str | None) -> str | None:
    """BibTeX writes page ranges with a double dash."""
    if not pages:
        return None
    return re.sub(r"\s*-\s*", "--", pages.strip())


def _note(paper: Paper) -> str | None:
    """Record provenance, which is otherwise lost on export."""
    sources = "+".join(source.value for source in paper.all_sources)
    return f"Retrieved via PaperPilot from {sources}"


def _ascii_only(value: str) -> str:
    """Fold accents away; citation keys must be plain ASCII."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(c for c in decomposed if not unicodedata.combining(c) and ord(c) < 128)
