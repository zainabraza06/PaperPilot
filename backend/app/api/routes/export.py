"""Citation export endpoints.

Two shapes, because the frontend needs both: a single paper exported from
a detail view, and a selection exported in bulk. Both return a downloadable
file rather than JSON, so the browser can save it without the client
having to reconstruct a blob.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app.api.deps import PaperStoreDep
from app.services.export import ExportFormat, get_formatter

router = APIRouter(prefix="/api/export", tags=["export"])

#: Bulk exports are bounded. An unbounded id list is a cheap way to make
#: the server read the entire store into memory.
_MAX_PAPERS = 500


class ExportRequest(BaseModel):
    """A bulk export of previously retrieved papers."""

    paper_ids: list[str] = Field(min_length=1, max_length=_MAX_PAPERS)
    format: ExportFormat = ExportFormat.BIBTEX
    filename: str | None = Field(
        default=None, description="Base name for the download, without extension."
    )


@router.post(
    "",
    summary="Export several papers as a citation file",
    response_class=Response,
)
async def export_many(request: ExportRequest, store: PaperStoreDep) -> Response:
    """Render the selected papers in the requested citation format.

    Papers are looked up from the store rather than accepted in the
    request body, so a citation can only be produced for a record the
    server actually retrieved.
    """
    papers = store.get_many(request.paper_ids)
    if not papers:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "none of the requested papers are known; run a search first",
        )

    formatter = get_formatter(request.format)
    body = formatter.format_many(papers)
    stem = _safe_stem(request.filename) or f"paperpilot-{len(papers)}-references"

    response = _download(body, formatter.media_type, f"{stem}.{formatter.extension}")
    # A partial selection is a real outcome: some ids may have aged out of
    # the store. Report it in a header rather than failing the whole export.
    if len(papers) < len(request.paper_ids):
        missing = len(request.paper_ids) - len(papers)
        response.headers["X-PaperPilot-Missing"] = str(missing)
    return response


@router.get(
    "/{paper_id}",
    summary="Export a single paper",
    response_class=Response,
)
async def export_one(
    paper_id: str,
    store: PaperStoreDep,
    format: ExportFormat = Query(default=ExportFormat.BIBTEX),
) -> Response:
    """Render one paper. A GET so it can be a plain link in the UI."""
    papers = store.get_many([paper_id])
    if not papers:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown paper {paper_id!r}")

    formatter = get_formatter(format)
    body = formatter.format_one(papers[0])
    stem = _safe_stem(papers[0].doi or paper_id) or "reference"
    return _download(body, formatter.media_type, f"{stem}.{formatter.extension}")


@router.get("/formats/available", summary="List the supported citation formats")
async def list_formats() -> list[dict[str, str]]:
    """Let the UI build its format picker without hardcoding the list."""
    labels = {
        ExportFormat.BIBTEX: "BibTeX",
        ExportFormat.RIS: "RIS (EndNote, Zotero, Mendeley)",
        ExportFormat.APA: "Plain text — APA 7th",
        ExportFormat.VANCOUVER: "Plain text — Vancouver",
    }
    return [
        {
            "id": fmt.value,
            "label": labels[fmt],
            "extension": get_formatter(fmt).extension,
        }
        for fmt in ExportFormat
    ]


def _download(body: str, media_type: str, filename: str) -> Response:
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _safe_stem(value: str | None) -> str:
    """Reduce a caller-supplied name to something safe in a header.

    A filename reaches the browser inside a Content-Disposition header, so
    quotes, newlines and path separators are all removed rather than
    escaped — there is no legitimate use for them here.
    """
    if not value:
        return ""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in value)
    return cleaned.strip("-.")[:80]
