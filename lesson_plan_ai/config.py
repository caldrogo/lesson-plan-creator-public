from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_model: str = "gemini-3.6-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "lesson_plan_evidence"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    max_revision_iterations: int = 2
    google_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
