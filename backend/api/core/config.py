"""
Application configuration — loaded from environment variables via pydantic-settings.
Copy .env.example → .env and fill in values before running.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the repo-root .env — main.py is documented to run with
# cwd=backend/, where a relative "env_file=.env" would silently resolve to a
# nonexistent backend/.env and load every value as empty.
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    # ----- API -----
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ----- CORS -----
    cors_origins: list[str] = ["http://localhost:4200"]

    # ----- Pinecone -----
    pinecone_api_key: str = ""
    pinecone_env: str = ""
    pinecone_index_name: str = "rag-finance-db"
    pinecone_namespace: str | None = None
    pinecone_upsert_batch_size: int = 100

    # ----- OpenAI / embeddings -----
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 100

    # ----- LLM -----
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ----- RAG tunables -----
    rag_top_k: int = 5
    # Calibrated against golden_dataset.jsonl (see
    # evaluation/calibrate_grounding_threshold.py) — correct/incorrect cosine
    # score distributions overlap substantially on this corpus (correct:
    # 0.379-0.778, incorrect: 0.387-0.729), so this midpoint is the best
    # available split, not a clean decision boundary. Revisit if the
    # orchestration layer adds LLM-judged groundedness instead of relying
    # solely on this score cutoff.
    grounding_threshold: float = 0.55
    hybrid_alpha: float = 0.5             # weight of dense in hybrid merge

    # ----- Chunking -----
    chunk_size: int = 250
    chunk_overlap: int = 50

    # ----- Memory / token budget -----
    sqlite_db_path: str = "./backend.db"
    token_budget_t: int = 3000            # unsummarised token limit
    token_budget_n: int = 5              # raw turns to keep after summarisation

    # extra="ignore": an unrecognized .env key (e.g. a typo'd var name) should
    # not hard-crash config loading for the whole app — it just won't populate
    # any field, which is easy to miss but far less disruptive than a boot crash.
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH), env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton per process)."""
    return Settings()
