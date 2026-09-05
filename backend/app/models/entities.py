"""Named entities extracted from a paper.

Spans are kept, not just surface forms, because the frontend highlights
entities inline in the abstract. A surface form can occur several times,
so one ``Entity`` carries every occurrence rather than being duplicated.

Offsets are relative to a named field (``title`` or ``abstract``) rather
than to a concatenation of the two, so the UI can highlight each field
independently without recomputing anything.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EntityLabel(str, Enum):
    """Entity categories, normalized across NER models.

    Different models emit wildly different label sets — ``en_core_web_sm``
    is trained on news (ORG, PERSON, GPE), while SciSpacy models emit
    biomedical types (DISEASE, CHEMICAL, GENE_OR_GENE_PRODUCT). Everything
    is mapped onto this enum so the API contract and the UI do not change
    when the underlying model does.
    """

    GENE_OR_PROTEIN = "gene_or_protein"
    DISEASE = "disease"
    CHEMICAL = "chemical"
    ORGANISM = "organism"
    ANATOMY = "anatomy"
    CELL_TYPE = "cell_type"
    #: Techniques, assays, algorithms, named models — detected by pattern
    #: rather than by a trained model when no scientific NER is installed.
    TECHNICAL_TERM = "technical_term"
    ORGANIZATION = "organization"
    PERSON = "person"
    LOCATION = "location"
    OTHER = "other"

    @property
    def is_scientific(self) -> bool:
        """True for the categories a researcher actually filters on."""
        return self in _SCIENTIFIC_LABELS


_SCIENTIFIC_LABELS = frozenset(
    {
        EntityLabel.GENE_OR_PROTEIN,
        EntityLabel.DISEASE,
        EntityLabel.CHEMICAL,
        EntityLabel.ORGANISM,
        EntityLabel.ANATOMY,
        EntityLabel.CELL_TYPE,
        EntityLabel.TECHNICAL_TERM,
    }
)

EntityField = Literal["title", "abstract"]


class EntitySpan(BaseModel):
    """One occurrence of an entity, as character offsets into a field."""

    model_config = {"frozen": True}

    field: EntityField
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class Entity(BaseModel):
    """A distinct entity mentioned in a paper, with all its occurrences."""

    text: str = Field(description="Surface form, as it appears in the text.")
    label: EntityLabel
    raw_label: str = Field(description="The label the underlying model emitted.")
    spans: list[EntitySpan] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.spans)

    @property
    def in_title(self) -> bool:
        """Entities mentioned in the title are the paper's actual subject."""
        return any(span.field == "title" for span in self.spans)
