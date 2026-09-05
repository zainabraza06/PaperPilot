"""Citation export tests.

The brief asks for each format's *actual* spec, not an approximation, so
the strongest tests here do not assert on strings we wrote — they feed our
output to an independent parser (``bibtexparser``, ``rispy``) and check
that another implementation reads back what we meant. Anything that only
we can parse is not really BibTeX.
"""

from __future__ import annotations

from datetime import date

import bibtexparser
import pytest
import rispy

from app.models.paper import Author, Paper, SourceName
from app.services.export import ExportFormat, get_formatter
from app.services.export.bibtex import BibTeXFormatter, escape, protect_case
from app.services.export.plaintext import PlainTextFormatter, TextStyle
from app.services.export.ris import RISFormatter


def make_paper(
    *,
    paper_id: str = "doi:10.1038/s41587-022-01234-5",
    title: str = "Prime editing efficiency in primary human cells",
    authors: list[Author] | None = None,
    doi: str | None = "10.1038/s41587-022-01234-5",
    journal: str | None = "Nature Biotechnology",
    published: date | None = date(2022, 7, 13),
    pages: str | None = "1021-1030",
    publication_type: str | None = "journal-article",
    source: SourceName = SourceName.PUBMED,
    **kwargs: object,
) -> Paper:
    return Paper(
        id=paper_id,
        doi=doi,
        source=source,
        source_id="35123456",
        title=title,
        authors=authors
        if authors is not None
        else [
            Author(name="Wei Chen", given="Wei", family="Chen"),
            Author(name="Adaeze Okafor", given="Adaeze", family="Okafor"),
        ],
        abstract="Prime editing enables precise genome edits.",
        published_date=published,
        url="https://doi.org/10.1038/s41587-022-01234-5",
        journal=journal,
        publisher="Springer",
        volume="40",
        issue="7",
        pages=pages,
        publication_type=publication_type,
        **kwargs,
    )


# --- BibTeX: round-tripped through an independent parser -----------------


def parse_bibtex(text: str) -> list[dict[str, str]]:
    return bibtexparser.loads(text).entries


def test_bibtex_output_parses_as_bibtex() -> None:
    entries = parse_bibtex(BibTeXFormatter().format_one(make_paper()))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["ENTRYTYPE"] == "article"
    assert entry["journal"] == "Nature Biotechnology"
    assert entry["doi"] == "10.1038/s41587-022-01234-5"
    assert entry["year"] == "2022"


def test_bibtex_author_names_round_trip_with_the_and_separator() -> None:
    # " and " is structural in BibTeX's name grammar, not punctuation: a
    # comma-separated list parses as one author with a very long surname.
    entry = parse_bibtex(BibTeXFormatter().format_one(make_paper()))[0]
    assert entry["author"] == "Chen, Wei and Okafor, Adaeze"


def test_bibtex_escapes_latex_control_characters() -> None:
    # An unescaped & breaks the *user's* build, which is the worst place
    # for our bug to surface.
    paper = make_paper(title="Cost & benefit: 50% of $X_1 in {vivo}", journal="Cell & Gene")
    rendered = BibTeXFormatter().format_one(paper)
    assert r"\&" in rendered
    assert r"\%" in rendered
    assert r"\$" in rendered
    assert r"\_" in rendered
    assert parse_bibtex(rendered)  # still parses


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a & b", r"a \& b"),
        ("100%", r"100\%"),
        ("$5", r"\$5"),
        ("a_b", r"a\_b"),
        ("#1", r"\#1"),
        ("{x}", r"\{x\}"),
        (r"back\slash", r"back\textbackslash{}slash"),
        ("~approx", r"\textasciitilde{}approx"),
        ("x^2", r"x\textasciicircum{}2"),
    ],
)
def test_every_latex_control_character_is_escaped(raw: str, expected: str) -> None:
    assert escape(raw) == expected


def test_backslash_is_escaped_before_the_characters_it_introduces() -> None:
    # Order matters: escaping & first would turn "&" into "\&" and then the
    # backslash pass would mangle it into "\textbackslash{}&".
    assert escape("a\\&b") == r"a\textbackslash{}\&b"


def test_acronyms_are_brace_protected_against_style_lowercasing() -> None:
    # Most BibTeX styles lowercase titles. Without protection "CRISPR-Cas9"
    # is typeset "Crispr-cas9", quietly corrupting every gene symbol.
    entry = parse_bibtex(
        BibTeXFormatter().format_one(make_paper(title="CRISPR-Cas9 editing in T cells"))
    )[0]
    assert "{CRISPR-Cas9}" in entry["title"]


def test_ordinary_words_are_not_brace_protected() -> None:
    # Protecting everything would fight the style rather than cooperate.
    assert protect_case("editing in human cells") == "editing in human cells"


@pytest.mark.parametrize(
    ("publication_type", "source", "journal", "expected"),
    [
        ("journal-article", SourceName.PUBMED, "Nature", "article"),
        ("proceedings-article", SourceName.CROSSREF, "NeurIPS", "inproceedings"),
        ("book-chapter", SourceName.CROSSREF, "A Book", "incollection"),
        ("dissertation", SourceName.CROSSREF, None, "phdthesis"),
        ("posted-content", SourceName.CROSSREF, None, "misc"),
        # An arXiv record with no journal is a preprint whatever it claims.
        (None, SourceName.ARXIV, None, "misc"),
        (None, SourceName.PUBMED, "Nature", "article"),
    ],
)
def test_entry_type_follows_the_record_not_a_default(
    publication_type: str | None, source: SourceName, journal: str | None, expected: str
) -> None:
    paper = make_paper(publication_type=publication_type, source=source, journal=journal)
    assert parse_bibtex(BibTeXFormatter().format_one(paper))[0]["ENTRYTYPE"] == expected


def test_citation_keys_are_unique_within_one_export() -> None:
    # Two papers by the same group in the same year is the normal case.
    papers = [make_paper(paper_id=f"p{n}", title="Prime editing outcomes") for n in range(3)]
    keys = [entry["ID"] for entry in parse_bibtex(BibTeXFormatter().format_many(papers))]
    assert len(set(keys)) == 3
    assert keys[0] == "chen_prime_2022"
    assert keys[1] == "chen_prime_2022a"


def test_citation_keys_are_ascii_even_for_accented_names() -> None:
    paper = make_paper(authors=[Author(name="Émile Zöller", given="Émile", family="Zöller")])
    key = parse_bibtex(BibTeXFormatter().format_one(paper))[0]["ID"]
    assert key.isascii()
    assert key.startswith("zoller")


def test_page_ranges_use_the_bibtex_double_dash() -> None:
    entry = parse_bibtex(BibTeXFormatter().format_one(make_paper()))[0]
    assert entry["pages"] == "1021--1030"


def test_consortium_authors_are_braced_so_they_are_not_split() -> None:
    # The braces are structural - they tell BibTeX this is one name, not
    # "Consortium, The Genome Editing". They must survive unescaped.
    paper = make_paper(authors=[Author(name="The Genome Editing Consortium")])
    rendered = BibTeXFormatter().format_one(paper)
    assert "author = {{The Genome Editing Consortium}}" in rendered
    assert r"\{The Genome" not in rendered


def test_author_names_are_escaped_exactly_once() -> None:
    # The author field escapes each name itself and then adds structural
    # braces, so the generic field path must not escape it a second time.
    paper = make_paper(authors=[Author(name="Smith & Sons", given="A", family="Smith & Sons")])
    rendered = BibTeXFormatter().format_one(paper)
    assert r"Smith \& Sons" in rendered
    assert "textbackslash" not in rendered


def test_a_paper_missing_everything_optional_still_produces_valid_bibtex() -> None:
    sparse = Paper(
        id="x",
        source=SourceName.ARXIV,
        source_id="1",
        title="A bare record",
        url="https://example.org/1",
    )
    entries = parse_bibtex(BibTeXFormatter().format_one(sparse))
    assert len(entries) == 1
    assert entries[0]["ENTRYTYPE"] == "misc"
    assert "nodate" in entries[0]["ID"]


def test_bulk_bibtex_parses_as_one_database() -> None:
    papers = [make_paper(paper_id=f"p{n}", title=f"Study number {n}") for n in range(5)]
    assert len(parse_bibtex(BibTeXFormatter().format_many(papers))) == 5


# --- RIS: round-tripped through an independent parser --------------------


def parse_ris(text: str) -> list[dict[str, object]]:
    return list(rispy.loads(text))


def test_ris_output_parses_as_ris() -> None:
    records = parse_ris(RISFormatter().format_one(make_paper()))
    assert len(records) == 1
    record = records[0]
    assert record["type_of_reference"] == "JOUR"
    assert record["title"] == "Prime editing efficiency in primary human cells"
    assert record["doi"] == "10.1038/s41587-022-01234-5"
    assert record["year"] == "2022"


def test_ris_tag_lines_use_exactly_two_spaces_before_the_hyphen() -> None:
    # One space instead of two and EndNote silently drops the field.
    rendered = RISFormatter().format_one(make_paper())
    for line in rendered.split("\r\n"):
        if not line.strip():
            continue
        assert line[2:6] == "  - ", f"malformed tag line: {line!r}"


def test_ris_records_start_with_ty_and_end_with_er() -> None:
    lines = [line for line in RISFormatter().format_one(make_paper()).split("\r\n") if line]
    assert lines[0].startswith("TY  - ")
    assert lines[-1] == "ER  - "


def test_ris_uses_crlf_line_endings_as_the_spec_requires() -> None:
    rendered = RISFormatter().format_one(make_paper())
    assert "\r\n" in rendered
    assert "\n" not in rendered.replace("\r\n", "")


def test_ris_repeats_the_tag_for_each_author() -> None:
    rendered = RISFormatter().format_one(make_paper())
    assert rendered.count("AU  - ") == 2
    assert parse_ris(rendered)[0]["authors"] == ["Chen, Wei", "Okafor, Adaeze"]


def test_ris_splits_page_ranges_into_start_and_end() -> None:
    record = parse_ris(RISFormatter().format_one(make_paper()))[0]
    assert record["start_page"] == "1021"
    assert record["end_page"] == "1030"


def test_ris_handles_a_single_page_with_no_end() -> None:
    record = parse_ris(RISFormatter().format_one(make_paper(pages="e1234")))[0]
    assert record["start_page"] == "e1234"
    assert "end_page" not in record


@pytest.mark.parametrize(
    ("publication_type", "source", "journal", "expected"),
    [
        ("journal-article", SourceName.PUBMED, "Nature", "JOUR"),
        ("proceedings-article", SourceName.CROSSREF, "NeurIPS", "CONF"),
        ("book-chapter", SourceName.CROSSREF, "A Book", "CHAP"),
        ("dissertation", SourceName.CROSSREF, None, "THES"),
        ("posted-content", SourceName.CROSSREF, None, "UNPB"),
        (None, SourceName.ARXIV, None, "UNPB"),
    ],
)
def test_ris_reference_type_follows_the_record(
    publication_type: str | None, source: SourceName, journal: str | None, expected: str
) -> None:
    paper = make_paper(publication_type=publication_type, source=source, journal=journal)
    assert parse_ris(RISFormatter().format_one(paper))[0]["type_of_reference"] == expected


def test_ris_collapses_newlines_that_would_be_read_as_new_tags() -> None:
    paper = make_paper(title="A title\nsplit over\nlines")
    record = parse_ris(RISFormatter().format_one(paper))[0]
    assert record["title"] == "A title split over lines"


def test_bulk_ris_parses_as_separate_records() -> None:
    papers = [make_paper(paper_id=f"p{n}", title=f"Study {n}") for n in range(4)]
    assert len(parse_ris(RISFormatter().format_many(papers))) == 4


# --- plain text -----------------------------------------------------------


def authors(count: int) -> list[Author]:
    return [
        Author(name=f"Given{n} Family{n}", given=f"Given{n}", family=f"Family{n}")
        for n in range(1, count + 1)
    ]


def test_apa_formats_a_standard_journal_article() -> None:
    citation = PlainTextFormatter(TextStyle.APA).format_one(make_paper())
    assert citation == (
        "Chen, W., & Okafor, A. (2022). Prime editing efficiency in primary human cells. "
        "Nature Biotechnology, 40(7), 1021-1030. "
        "https://doi.org/10.1038/s41587-022-01234-5"
    )


def test_apa_uses_an_ampersand_before_the_final_author() -> None:
    citation = PlainTextFormatter(TextStyle.APA).format_one(make_paper(authors=authors(3)))
    assert "Family1, G., Family2, G., & Family3, G." in citation


def test_apa_lists_up_to_twenty_authors_in_full() -> None:
    citation = PlainTextFormatter(TextStyle.APA).format_one(make_paper(authors=authors(20)))
    assert "..." not in citation
    assert "Family20, G." in citation


def test_apa_with_more_than_twenty_authors_ends_with_the_last_not_the_twentieth() -> None:
    # The common bug: APA 7th wants the first 19, an ellipsis, then the
    # *final* author, not the twentieth.
    citation = PlainTextFormatter(TextStyle.APA).format_one(make_paper(authors=authors(25)))
    assert "Family19, G., ... Family25, G." in citation
    assert "Family20" not in citation


def test_apa_uses_n_d_when_there_is_no_date() -> None:
    citation = PlainTextFormatter(TextStyle.APA).format_one(make_paper(published=None))
    assert "(n.d.)" in citation


def test_vancouver_lists_six_authors_then_et_al() -> None:
    citation = PlainTextFormatter(TextStyle.VANCOUVER).format_one(make_paper(authors=authors(8)))
    assert "Family6 G, et al." in citation
    assert "Family7" not in citation


def test_vancouver_uses_no_periods_between_surname_and_initials() -> None:
    citation = PlainTextFormatter(TextStyle.VANCOUVER).format_one(make_paper())
    assert citation.startswith("Chen W, Okafor A.")


def test_vancouver_uses_semicolon_and_colon_volume_page_punctuation() -> None:
    citation = PlainTextFormatter(TextStyle.VANCOUVER).format_one(make_paper())
    assert "Nature Biotechnology. 2022;40(7):1021-1030." in citation


def test_vancouver_bulk_export_is_numbered() -> None:
    papers = [make_paper(paper_id=f"p{n}", title=f"Study {n}") for n in range(3)]
    rendered = PlainTextFormatter(TextStyle.VANCOUVER).format_many(papers)
    assert rendered.startswith("1. ")
    assert "\n2. " in rendered


def test_plain_text_survives_a_paper_with_no_authors_or_venue() -> None:
    sparse = Paper(
        id="x",
        source=SourceName.ARXIV,
        source_id="1",
        title="A bare record",
        url="https://example.org/1",
    )
    for style in TextStyle:
        citation = PlainTextFormatter(style).format_one(sparse)
        assert "A bare record" in citation
        assert "None" not in citation


# --- the registry ---------------------------------------------------------


@pytest.mark.parametrize("export_format", list(ExportFormat))
def test_every_declared_format_produces_output(export_format: ExportFormat) -> None:
    formatter = get_formatter(export_format)
    rendered = formatter.format_many([make_paper()])
    assert rendered.strip()
    assert formatter.extension
    assert formatter.media_type


def test_exporting_no_papers_is_not_an_error() -> None:
    for export_format in ExportFormat:
        assert get_formatter(export_format).format_many([]) == ""


def test_trailing_punctuation_stays_outside_the_protective_braces() -> None:
    # "{DNA}." not "{DNA.}" - the braces protect capitalisation, and a full
    # stop has no capitalisation to protect.
    assert protect_case("editing without donor DNA.") == "editing without donor {DNA}."
    assert protect_case("using CRISPR, then PCR;") == "using {CRISPR}, then {PCR};"
