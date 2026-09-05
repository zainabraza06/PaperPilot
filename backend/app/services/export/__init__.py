"""Citation export in BibTeX, RIS and plain text.

One registry, so a route never names a concrete formatter and adding a
format (CSL-JSON, EndNote XML) is a new module plus one entry here.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from app.models.paper import Paper
from app.services.export.bibtex import BibTeXFormatter
from app.services.export.plaintext import PlainTextFormatter, TextStyle
from app.services.export.ris import RISFormatter


class ExportFormat(str, Enum):
    """Formats a user can ask for.

    APA and Vancouver are separate members rather than a style parameter
    on one member: from the caller's point of view they are two different
    things to click, and modelling them as one format with a modifier
    pushes that distinction into every caller.
    """

    BIBTEX = "bibtex"
    RIS = "ris"
    APA = "apa"
    VANCOUVER = "vancouver"


@runtime_checkable
class CitationFormatter(Protocol):
    """Turns papers into a citation document."""

    #: Stable identifier, also used in the downloaded filename.
    format_id: str
    #: File extension, without the dot.
    extension: str
    #: MIME type for the HTTP response.
    media_type: str

    def format_one(self, paper: Paper) -> str: ...

    def format_many(self, papers: list[Paper]) -> str: ...


_FORMATTERS: dict[ExportFormat, CitationFormatter] = {
    ExportFormat.BIBTEX: BibTeXFormatter(),
    ExportFormat.RIS: RISFormatter(),
    ExportFormat.APA: PlainTextFormatter(TextStyle.APA),
    ExportFormat.VANCOUVER: PlainTextFormatter(TextStyle.VANCOUVER),
}


def get_formatter(export_format: ExportFormat) -> CitationFormatter:
    """Return the formatter for a requested format."""
    return _FORMATTERS[export_format]


__all__ = [
    "BibTeXFormatter",
    "CitationFormatter",
    "ExportFormat",
    "PlainTextFormatter",
    "RISFormatter",
    "TextStyle",
    "get_formatter",
]
