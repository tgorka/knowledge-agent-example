"""Typed configuration loaded from `.env`. One source of truth for credentials."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "knowledgeagent"
    neo4j_database: str = "neo4j"

    openrouter_api_key: str = "sk-or-v1-REPLACE-ME"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-5.5"
    openrouter_app_name: str = "knowledge-agent-example"
    openrouter_http_referer: str = "http://localhost"

    filestore_dir: Path = Field(default=Path("./filestore"))
    filestore_bucket: str = "local-prov-store"

    schema_evolution_require_approval: bool = True
    agent_verbosity: int = 1

    # Cache the live Neo4j schema for this many seconds before re-querying
    # the database. Schema evolution explicitly invalidates the cache.
    schema_cache_ttl_seconds: int = 30
    # Number of automatic retries on 429 / 5xx from the LLM gateway.
    llm_max_retries: int = 3
    # Default completion-length cap. 1024 is enough for a Cypher query +
    # one-paragraph summary and fits inside OpenRouter free-tier per-request
    # limits.
    llm_max_tokens: int = 1024
    # Schema-evolution prompts emit larger structured JSON (proposal +
    # per-payload extraction). 2048 is comfortable; raise to 4096 for
    # heavier payload files.
    llm_max_tokens_structured: int = 2048


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.filestore_dir.mkdir(parents=True, exist_ok=True)
    return _settings
