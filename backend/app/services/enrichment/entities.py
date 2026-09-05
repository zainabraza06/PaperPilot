"""Named-entity extraction over titles and abstracts.

The honest framing matters here, because it drives the design:

A general-purpose NER model — ``en_core_web_sm``, trained on news — is
close to useless on a scientific abstract *for the entities a researcher
cares about*. It reliably tags institutions and places, and it will
happily label "CRISPR" as an ORG and every number as a CARDINAL. The
high-value entities in an abstract are gene symbols, assays, techniques
and named models, and a news-trained model has never seen them as a
category.

So this module runs two passes and says which produced what:

1. **Model NER** — whatever spaCy pipeline is installed, with its labels
   normalized onto :class:`EntityLabel` and the purely numeric categories
   discarded. If a SciSpacy model is installed it is preferred, and this
   pass becomes genuinely good: typed diseases, chemicals and genes.
2. **Pattern extraction** — acronyms and symbol-shaped tokens
   (``pegRNA``, ``CRISPR-Cas9``, ``BRCA1``, ``scRNA-seq``, ``PE3``). These
   are what the general model misses, they are trivially detectable by
   shape, and they are the terms a researcher actually scans for.

The second pass only fills spans the first did not claim, so the frontend
never has to resolve overlapping highlights.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from app.core.logging import get_logger
from app.models.entities import Entity, EntityField, EntityLabel, EntitySpan
from app.models.paper import Paper

logger = get_logger(__name__)

# Preferred first: SciSpacy models emit typed biomedical entities, which is
# what this pipeline actually wants. The general model is the fallback.
_MODEL_PREFERENCE = (
    "en_ner_bc5cdr_md",  # SciSpacy: diseases + chemicals
    "en_ner_bionlp13cg_md",  # SciSpacy: genes, cells, organisms, tissues
    "en_core_sci_sm",  # SciSpacy: untyped scientific entities
    "en_core_web_sm",  # spaCy general-purpose (news)
)

#: Raw model labels -> our normalized vocabulary.
_LABEL_MAP: dict[str, EntityLabel] = {
    # --- SciSpacy: bc5cdr, bionlp13cg, craft ---
    "DISEASE": EntityLabel.DISEASE,
    "CANCER": EntityLabel.DISEASE,
    "CHEMICAL": EntityLabel.CHEMICAL,
    "SIMPLE_CHEMICAL": EntityLabel.CHEMICAL,
    "GENE_OR_GENE_PRODUCT": EntityLabel.GENE_OR_PROTEIN,
    "AMINO_ACID": EntityLabel.GENE_OR_PROTEIN,
    "PROTEIN": EntityLabel.GENE_OR_PROTEIN,
    "GGP": EntityLabel.GENE_OR_PROTEIN,
    "ORGANISM": EntityLabel.ORGANISM,
    "TAXON": EntityLabel.ORGANISM,
    "ORGANISM_SUBDIVISION": EntityLabel.ANATOMY,
    "ANATOMICAL_SYSTEM": EntityLabel.ANATOMY,
    "TISSUE": EntityLabel.ANATOMY,
    "ORGAN": EntityLabel.ANATOMY,
    "CELL": EntityLabel.CELL_TYPE,
    "CELL_TYPE": EntityLabel.CELL_TYPE,
    "CELL_LINE": EntityLabel.CELL_TYPE,
    "CELLULAR_COMPONENT": EntityLabel.CELL_TYPE,
    "ENTITY": EntityLabel.OTHER,  # en_core_sci_sm is untyped
    # --- spaCy general-purpose ---
    "ORG": EntityLabel.ORGANIZATION,
    "PERSON": EntityLabel.PERSON,
    "GPE": EntityLabel.LOCATION,
    "LOC": EntityLabel.LOCATION,
    "NORP": EntityLabel.OTHER,
    "PRODUCT": EntityLabel.TECHNICAL_TERM,
    "FAC": EntityLabel.LOCATION,
}

#: SciSpacy pipelines, whose labels are trusted in full because they were
#: trained on exactly this text. Anything else is treated as a general model.
_SCIENTIFIC_MODELS = frozenset(
    {"en_ner_bc5cdr_md", "en_ner_bionlp13cg_md", "en_ner_craft_md", "en_ner_jnlpba_md",
     "en_core_sci_sm", "en_core_sci_md", "en_core_sci_lg", "en_core_sci_scibert"}
)

#: The only labels a *general* (news-trained) model is trusted for on
#: scientific text, and even these are guarded — see `_trust_general`.
#: Measured, not assumed. On real abstracts en_core_web_sm produced
#: ("pegRNA improves CRISPR-Cas9", LAW), ("HEK293", GPE), and ORG for each
#: of "CRISPR-Cas9", "FAB-CRISPR" and "HDR": a technique read as
#: legislation, a cell line read as a country, and assays read as
#: companies. The one correct ORG in the sample was "the Broad Institute".
#: So the model contributes institutions and people only, and only where
#: the span does not look like a technical term.
_GENERAL_MODEL_TRUSTED = frozenset({"ORG", "PERSON"})

#: Words that make a single-token ORG believable in a scientific abstract.
_ORG_SUFFIXES = (
    "institute",
    "university",
    "hospital",
    "laboratory",
    "foundation",
    "college",
    "school",
    "centre",
    "center",
    "consortium",
    "agency",
    "ministry",
    "inc",
    "ltd",
    "gmbh",
)

#: Dropped outright. Dates, counts and percentages are abundant in abstracts
#: and carry no value as filterable entities — they would drown the useful
#: ones in the UI.
_DISCARDED_LABELS = frozenset(
    {
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
        "LANGUAGE",
        "WORK_OF_ART",
        "LAW",
        "EVENT",
    }
)

# --- pattern pass --------------------------------------------------------

#: Uppercase acronyms and symbols: DNA, CRISPR, PE3, BRCA1, SARS-CoV-2.
_ACRONYM = re.compile(r"\b[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
#: Lowercase-prefixed symbols: pegRNA, mRNA, sgRNA, scRNA, siRNA.
_PREFIXED_SYMBOL = re.compile(r"\b[a-z]{1,4}[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
#: Assay names: RNA-seq, ChIP-seq, scRNA-seq, ATAC-seq.
_SEQ_ASSAY = re.compile(r"\b[A-Za-z][A-Za-z0-9]*-[Ss]eq\b")

#: Section headings from structured abstracts match the acronym shape and
#: are not entities. PubMed abstracts are re-assembled with these labels.
_SECTION_LABELS = frozenset(
    """
    ABSTRACT AIM AIMS BACKGROUND CONCLUSION CONCLUSIONS DESIGN DISCUSSION
    FINDINGS FUNDING IMPORTANCE INTERPRETATION INTERVENTION INTRODUCTION
    LIMITATIONS METHOD METHODS OBJECTIVE OBJECTIVES OUTCOME OUTCOMES
    PARTICIPANTS PATIENTS PURPOSE RATIONALE RESULT RESULTS SETTING
    SIGNIFICANCE SUMMARY
    """.split()
)

#: Common English words and units that match an acronym shape.
_PATTERN_STOPWORDS = frozenset(
    {"I", "A", "AND", "OR", "THE", "IN", "OF", "WE", "US", "IT", "AN", "AS", "AT", "BY"}
)


@runtime_checkable
class EntityExtractor(Protocol):
    """Extracts entities from a batch of papers."""

    #: Names the pipeline that produced the entities, for attribution.
    model_id: str

    def extract(self, papers: Sequence[Paper]) -> list[list[Entity]]:
        """Return one entity list per paper, in the same order."""
        ...


class SpacyEntityExtractor:
    """spaCy-backed extraction, with a shape-based pass for science terms."""

    def __init__(self, model_name: str | None = None, *, patterns: bool = True) -> None:
        self._requested = model_name
        self._model_name: str | None = None
        self._nlp: object | None = None
        self._lock = threading.Lock()
        self._patterns = patterns
        self._scientific = False
        self.model_id = model_name or "spacy:auto"

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        with self._lock:
            if self._nlp is not None:
                return
            import spacy

            candidates = (self._requested,) if self._requested else _MODEL_PREFERENCE
            errors: list[str] = []
            for name in candidates:
                if not name:
                    continue
                try:
                    # The parser and lemmatizer cost real time and nothing
                    # here reads their output.
                    nlp = spacy.load(name, exclude=["parser", "lemmatizer", "textcat"])
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                    continue
                self._nlp = nlp
                self._model_name = name
                self._scientific = name in _SCIENTIFIC_MODELS
                self.model_id = name
                logger.info(
                    "entity model ready: %s (%s)",
                    name,
                    "scientific, labels trusted"
                    if self._scientific
                    else "general-purpose, only ORG/PERSON trusted",
                )
                return
            raise RuntimeError("no spaCy model could be loaded: " + "; ".join(errors))

    def extract(self, papers: Sequence[Paper]) -> list[list[Entity]]:
        if not papers:
            return []
        self._ensure_loaded()
        nlp = self._nlp
        assert nlp is not None

        # One flat batch across both fields of every paper: spaCy's pipe is
        # much faster batched, and per-document calls would waste that.
        jobs: list[tuple[int, EntityField, str]] = []
        for index, paper in enumerate(papers):
            jobs.append((index, "title", paper.title))
            if paper.abstract:
                jobs.append((index, "abstract", paper.abstract))

        collected: list[list[tuple[str, EntityLabel, str, EntitySpan]]] = [[] for _ in papers]
        docs = nlp.pipe([text for _, _, text in jobs], batch_size=32)  # type: ignore[attr-defined]

        for (index, field, text), doc in zip(jobs, docs, strict=True):
            found = list(self._from_model(doc, field, self._scientific))
            claimed = {(span.start, span.end) for _, _, _, span in found}
            if self._patterns:
                found.extend(self._from_patterns(text, field, claimed))
            collected[index].extend(found)

        return [_merge_occurrences(items) for items in collected]

    @staticmethod
    def _from_model(
        doc: object, field: EntityField, scientific: bool
    ) -> Iterable[tuple[str, EntityLabel, str, EntitySpan]]:
        """Yield the model's entities, scoped to the labels it can be trusted for."""
        for ent in doc.ents:  # type: ignore[attr-defined]
            raw = str(ent.label_)
            if raw in _DISCARDED_LABELS:
                continue
            if not scientific and not _trust_general(raw, ent.text, field):
                continue
            start = int(ent.start_char)
            text = ent.text
            # Leading articles come along with ORG spans ("the Broad
            # Institute") and are noise in a filter chip.
            for article in ("the ", "The ", "a ", "A ", "an ", "An "):
                if text.startswith(article):
                    start += len(article)
                    text = text[len(article) :]
                    break
            text = text.strip()
            if len(text) < 2 or text.upper() in _SECTION_LABELS:
                continue
            label = _LABEL_MAP.get(raw, EntityLabel.OTHER)
            yield text, label, raw, EntitySpan(field=field, start=start, end=start + len(text))

    @staticmethod
    def _from_patterns(
        text: str, field: EntityField, claimed: set[tuple[int, int]]
    ) -> list[tuple[str, EntityLabel, str, EntitySpan]]:
        """Find symbol-shaped technical terms the model did not claim."""
        found: list[tuple[str, EntityLabel, str, EntitySpan]] = []
        taken = set(claimed)
        for pattern in (_SEQ_ASSAY, _ACRONYM, _PREFIXED_SYMBOL):
            for match in pattern.finditer(text):
                surface = match.group(0)
                if surface.upper() in _SECTION_LABELS or surface.upper() in _PATTERN_STOPWORDS:
                    continue
                span = (match.start(), match.end())
                if any(span[0] < end and start < span[1] for start, end in taken):
                    continue  # already highlighted by an earlier pass
                taken.add(span)
                found.append(
                    (
                        surface,
                        EntityLabel.TECHNICAL_TERM,
                        "pattern",
                        EntitySpan(field=field, start=span[0], end=span[1]),
                    )
                )
        return found


class PatternEntityExtractor:
    """Shape-based extraction only, for when no spaCy model is installed.

    Strictly weaker — no organizations, no diseases, no typed biomedical
    entities — but it still surfaces the acronyms and gene symbols that
    dominate an abstract, so the UI has something real to highlight rather
    than an empty panel. ``model_id`` makes the downgrade visible.
    """

    model_id = "pattern-only"

    def extract(self, papers: Sequence[Paper]) -> list[list[Entity]]:
        results: list[list[Entity]] = []
        for paper in papers:
            found: list[tuple[str, EntityLabel, str, EntitySpan]] = []
            found.extend(SpacyEntityExtractor._from_patterns(paper.title, "title", set()))
            if paper.abstract:
                found.extend(
                    SpacyEntityExtractor._from_patterns(paper.abstract, "abstract", set())
                )
            results.append(_merge_occurrences(found))
        return results


def _trust_general(raw_label: str, text: str, field: EntityField) -> bool:
    """Decide whether to believe a general-purpose model on this span.

    Three guards, every one of them written in response to an observed
    failure rather than out of caution:

    * A span shaped like a technical term *is* a technical term. The model
      calls "CRISPR-Cas9" and "HDR" organizations; the pattern pass labels
      them correctly once the model stops claiming the span.
    * Titles are title-cased, which a news-trained model reads as evidence
      of a proper noun — that is how "Fast Antibiotic" became an ORG. Its
      title predictions are discarded outright.
    * A single-word ORG in a scientific abstract is almost always a method
      or a tool, not an institution ("Transformer", "BERT"). Real
      institutions are multi-word or carry an institutional suffix.
    """
    if raw_label not in _GENERAL_MODEL_TRUSTED:
        return False
    if field == "title":
        return False
    stripped = text.strip()
    if (
        _ACRONYM.fullmatch(stripped)
        or _PREFIXED_SYMBOL.fullmatch(stripped)
        or _SEQ_ASSAY.fullmatch(stripped)
    ):
        return False
    if raw_label == "ORG" and len(stripped.split()) == 1:
        return stripped.lower().endswith(_ORG_SUFFIXES)
    return True


def _merge_occurrences(
    items: Sequence[tuple[str, EntityLabel, str, EntitySpan]],
) -> list[Entity]:
    """Collapse repeated mentions into one entity carrying every span.

    Grouped case-insensitively, because "CRISPR" and "Crispr" in the same
    abstract are one entity to a reader.

    A surface form gets exactly one label. Grouping by (text, label) instead
    let "PE" appear twice in one paper — as a technical term seven times and
    an organization six — which is noise in a filter list and a bug in a
    highlight layer. When passes disagree, the scientific label wins,
    because that is the pass that is right about this text.
    """
    grouped: dict[str, list[tuple[str, EntityLabel, str, EntitySpan]]] = {}
    for text, label, raw_label, span in items:
        grouped.setdefault(text.lower(), []).append((text, label, raw_label, span))

    entities: list[Entity] = []
    for occurrences in grouped.values():
        label, raw_label = _dominant_label(occurrences)
        surfaces = [surface for surface, _, _, _ in occurrences]
        display = max(set(surfaces), key=surfaces.count)
        spans = sorted(
            {span for _, _, _, span in occurrences}, key=lambda s: (s.field, s.start)
        )
        entities.append(
            Entity(text=display, label=label, raw_label=raw_label, spans=spans)
        )

    entities.sort(key=lambda e: (e.in_title, e.count, e.label.is_scientific), reverse=True)
    return entities


def _dominant_label(
    occurrences: Sequence[tuple[str, EntityLabel, str, EntitySpan]],
) -> tuple[EntityLabel, str]:
    """Pick the single label for a surface form that two passes disagreed on."""
    tally: dict[EntityLabel, tuple[int, str]] = {}
    for _, label, raw_label, _ in occurrences:
        count, existing_raw = tally.get(label, (0, raw_label))
        tally[label] = (count + 1, existing_raw)
    label = max(tally, key=lambda item: (item.is_scientific, tally[item][0], item.value))
    return label, tally[label][1]


def build_entity_extractor(model_name: str | None) -> EntityExtractor:
    """Return the configured extractor, degrading if spaCy is unavailable.

    A missing NER model must not take search down: entities are an
    enrichment, and the pattern-only extractor still produces something
    useful. The degradation is reported through ``model_id``.
    """
    extractor = SpacyEntityExtractor(model_name)
    try:
        extractor._ensure_loaded()
    except Exception as exc:
        logger.warning(
            "no spaCy NER model available (%s); falling back to pattern extraction", exc
        )
        return PatternEntityExtractor()
    return extractor
