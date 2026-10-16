from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Deliberately no teacher API key and no .env loading in this contract.
    model_config = SettingsConfigDict(env_prefix="LP_", extra="ignore")
    generation_model: str = "gemini-3.5-flash-lite"
    default_duration: int = Field(default=60, ge=10, le=240)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    embedding_dimension: int = 384
    embedding_batch_size: int = Field(default=8, ge=1, le=64)
    ingestion_batch_size: int = Field(default=8, ge=1, le=256)
    collection: str = "lesson_plan_v2"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    corpus_directory: str = "data/processed/v2"
    timeout_seconds: int = Field(default=300, ge=1)
