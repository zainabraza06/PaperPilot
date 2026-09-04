"""Health and capability endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import RegistryDep, SettingsDep

router = APIRouter(tags=["system"])


class SourceInfo(BaseModel):
    name: str
    display_name: str


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    sources: list[SourceInfo]


@router.get("/health", response_model=HealthResponse, summary="Liveness and configured sources")
async def health(settings: SettingsDep, registry: RegistryDep) -> HealthResponse:
    """Report liveness plus which connectors this instance has registered.

    The source list is part of the health payload because the frontend uses
    it to render source filters without hardcoding provider names.
    """
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
        sources=[
            SourceInfo(name=source.name.value, display_name=source.display_name)
            for source in registry.all()
        ],
    )
