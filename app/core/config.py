
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Enterprise-IT-Support-Agentic-RAG-Copilot"
    app_env: str = "development"

    # AI model configuration
    llm_provider: str = "ollama"
    llm_request_timeout_seconds: float = 35.0
    tavily_request_timeout_seconds: float = 10.0
    hf_hub_etag_timeout_seconds: float = 10.0
    hf_hub_download_timeout_seconds: float = 30.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    pinecone_api_key: str = ""
    tavily_api_key: str = ""

    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_model_path: str = str(
        Path.home() / ".cache" / "enterprise-it-support-fastembed"
    )
    embedding_local_files_only: bool = True
    pinecone_control_timeout_seconds: float = 8.0
    pinecone_query_timeout_seconds: float = 15.0
    pinecone_index_ready_timeout_seconds: float = 60.0

    top_k: int = 4
    max_retries: int = 1

    admin_api_key: str = "change-me"

    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
