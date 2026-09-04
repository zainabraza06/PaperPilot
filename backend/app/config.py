"""Application settings, loaded from the environment (and ``.env`` if present)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Every knob that differs between a laptop and a container lives here.
    Defaults are chosen so the app runs with no configuration at all.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PAPERPILOT_",
        extra="ignore",
    )

    app_name: str = "PaperPilot"
    environment: str = "development"
    log_level: str = "INFO"

    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Origins allowed to call the API from a browser.",
    )

    # --- shared HTTP behaviour ------------------------------------------
    http_timeout_seconds: float = 15.0
    source_timeout_seconds: float = Field(
        default=20.0,
        description="Wall-clock budget for one source's entire search, retries included.",
    )
    http_max_retries: int = 2
    user_agent: str = Field(
        default="PaperPilot/0.1 (https://github.com/zainabraza06/PaperPilot)",
        description="Sent on every outbound request; all three APIs ask for identification.",
    )

    # --- provider specifics ----------------------------------------------
    # NCBI allows 3 req/s anonymously and 10 req/s with an API key.
    pubmed_api_key: str | None = None
    pubmed_email: str | None = None
    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # arXiv's terms of use ask for one request every three seconds.
    arxiv_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_min_interval_seconds: float = 3.0

    # Supplying a mailto puts us in Crossref's faster "polite" pool.
    crossref_base_url: str = "https://api.crossref.org"
    crossref_mailto: str | None = None

    # --- search defaults --------------------------------------------------
    default_results_per_source: int = 20
    max_results_per_source: int = 100

    # --- ranking (Stage 2) -------------------------------------------------
    ranking_enabled: bool = True
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "sentence-transformers model id. Set empty to force the offline "
            "hashing fallback, which ranks noticeably worse."
        ),
    )
    ranking_strategy: str = Field(
        default="linear",
        description="Fusion strategy: semantic | lexical | linear | rrf.",
    )
    ranking_alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight on the semantic signal in linear fusion; lexical gets 1 - alpha.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
