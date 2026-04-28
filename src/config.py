"""Environment and runtime configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # The application can still run in hosted environments that inject env vars.
    pass


MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@dataclass(frozen=True)
class AppConfig:
    pinecone_api_key: Optional[str]
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float
    llm_provider: str
    llm_model: str
    groq_api_key: Optional[str]
    openai_api_key: Optional[str]
    llm_api_key: Optional[str]
    llm_base_url: Optional[str]
    query_log_path: str


def _get_int(name: str, default: int) -> int:
    value = _get_setting(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = _get_setting(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def sanitize_index_name(value: str) -> str:
    """Return a Pinecone-compatible index name."""
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    if not normalized:
        return "docuquery-rag"
    return normalized[:45].strip("-") or "docuquery-rag"


def sanitize_namespace(value: str) -> str:
    normalized = re.sub(r"\s+", "-", value.strip())
    return normalized or "default"


def _get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value is not None:
            return str(secret_value)
    except Exception:
        pass

    return default


def get_config() -> AppConfig:
    index_name = sanitize_index_name(_get_setting("PINECONE_INDEX_NAME", "docuquery-rag") or "")
    namespace = sanitize_namespace(_get_setting("PINECONE_NAMESPACE", "course-assignment") or "")
    chunk_size = _get_int("CHUNK_SIZE", 500)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 80)
    top_k = _get_int("TOP_K", 8)
    threshold = _get_float("SIMILARITY_THRESHOLD", 0.25)
    llm_provider = (_get_setting("LLM_PROVIDER", "groq") or "groq").lower()
    llm_model = _get_setting("LLM_MODEL")
    if llm_provider == "groq":
        llm_model = _get_setting("GROQ_MODEL") or llm_model

    return AppConfig(
        pinecone_api_key=_get_setting("PINECONE_API_KEY"),
        pinecone_index_name=index_name,
        pinecone_cloud=_get_setting("PINECONE_CLOUD", "aws") or "aws",
        pinecone_region=_get_setting("PINECONE_REGION", "us-east-1") or "us-east-1",
        pinecone_namespace=namespace,
        embedding_model_name=_get_setting(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        or "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=max(100, chunk_size),
        chunk_overlap=max(0, chunk_overlap),
        top_k=max(1, top_k),
        similarity_threshold=min(max(threshold, 0.0), 1.0),
        llm_provider=llm_provider,
        llm_model=llm_model or "llama-3.3-70b-versatile",
        groq_api_key=_get_setting("GROQ_API_KEY"),
        openai_api_key=_get_setting("OPENAI_API_KEY"),
        llm_api_key=_get_setting("LLM_API_KEY"),
        llm_base_url=_get_setting("LLM_BASE_URL"),
        query_log_path=_get_setting("QUERY_LOG_PATH", "logs/query_log.csv")
        or "logs/query_log.csv",
    )


def mask_secret(value: Optional[str]) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "set"
    return f"{value[:4]}...{value[-4:]}"
