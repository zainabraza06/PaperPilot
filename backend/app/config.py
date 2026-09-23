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
        default=12.0,
        description=(
            "Wall-clock budget for one source's entire search, retries included. "
            "A hung source costs the user exactly this much, since the fan-out can "
            "only return once its slowest branch resolves. Measured floor: PubMed's "
            "two-call esearch/efetch pattern has been seen to take 10.7s, so this "
            "cannot go much lower without cutting off a healthy source."
        ),
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
    # --- enrichment: NER and clustering (Stage 3) --------------------------
    entities_enabled: bool = True
    ner_model: str | None = Field(
        default=None,
        description=(
            "spaCy model id. Left empty, the best installed model is chosen: "
            "a SciSpacy biomedical model if present, else en_core_web_sm, else "
            "pattern-only extraction."
        ),
    )
    clustering_enabled: bool = True
    max_clusters: int = Field(default=6, ge=2, le=20)
    min_papers_to_cluster: int = Field(default=8, ge=2)

    # --- summarization (Stage 4) -------------------------------------------
    summaries_enabled: bool = True
    llm_provider: str = Field(
        default="mistral",
        description="LLM provider for summaries: mistral | none.",
    )
    mistral_api_key: str | None = Field(
        default=None,
        description=(
            "Without a key the app still runs: summaries fall back to "
            "extractive sentences taken verbatim from the abstract."
        ),
    )
    summary_model: str = Field(
        default="ministral-8b-latest",
        description=(
            "Mistral quota is allocated per model, not per account: a valid key can "
            "have zero allowance on one model and hundreds of requests/minute on "
            "another. ministral-8b-latest is the default for its 188 req/min "
            "headroom, which covers a full result set without throttling."
        ),
    )
    summary_max_concurrent: int = Field(default=5, ge=1, le=20)
    summary_max_attempts: int = Field(
        default=2, ge=1, le=4, description="Generation attempts before falling back."
    )
    grounding_min_overlap: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum share of a summary's content words that must appear in the "
            "abstract. Measured, not guessed: 0.45 is the lowest value that still "
            "detects 100% of wrong-paper drift on the labelled set, and it lifts "
            "live first-attempt acceptance from 50% to 80%."
        ),
    )
    paper_store_path: str = Field(
        default="data/paperpilot.db",
        description=(
            "SQLite file holding retrieved papers so they can be exported "
            "later. Empty disables storage, and with it citation export."
        ),
    )
    grounding_min_support: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity between a summary sentence and its best "
            "matching abstract sentence. This is the only non-lexical grounding "
            "rule and the only one that can see a recombination - a fluent claim "
            "built from the abstract's own words. Set 0 to disable it."
        ),
    )
    summary_cache_path: str = Field(
        default="data/paperpilot.db",
        description="SQLite file for cached summaries. Empty disables caching.",
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
