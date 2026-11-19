"""Configuration. Environment variables first, Streamlit secrets second, safe defaults last."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=False)

APP_NAME = "Smart RAG Chatbot"
APP_ICON = "🤖"
APP_VERSION = "3.0.0"
TAGLINE = "Ask questions. Get cited answers."

# Models are discovered from the provider at run time; these are only the fallback list.
GROQ_FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GROQ_PREFERRED = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GROQ_RATE_LIMIT_FALLBACK = "llama-3.1-8b-instant"


def _secret(name: str) -> str | None:
    """Read a Streamlit secret without crashing when no secrets file exists."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _get(name: str, default: str = "") -> str:
    return os.getenv(name) or _secret(name) or default


@dataclass(frozen=True)
class Settings:
    provider: str            # "groq", "ollama" or "offline"
    groq_api_key: str | None
    groq_model: str
    ollama_host: str
    ollama_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_file_mb: int
    max_files: int
    max_chunks: int
    memory_turns: int
    temperature: float
    max_tokens: int

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


def load_settings() -> Settings:
    key = _get("GROQ_API_KEY") or None
    requested = _get("LLM_PROVIDER").lower()
    if requested in ("groq", "ollama", "offline"):
        provider = requested
    else:
        provider = "groq" if key else "offline"
    if provider == "groq" and not key:
        provider = "offline"          # never crash for want of a key; degrade instead

    def num(name: str, default: str, cast=int):
        try:
            return cast(_get(name, default))
        except ValueError:
            return cast(default)

    chunk_size = max(200, num("CHUNK_SIZE", "900"))
    overlap = min(num("CHUNK_OVERLAP", "150"), chunk_size // 2)
    return Settings(
        provider=provider,
        groq_api_key=key,
        groq_model=_get("GROQ_MODEL", GROQ_PREFERRED[0]),
        ollama_host=_get("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=_get("OLLAMA_MODEL", "mistral"),
        embedding_model=_get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=min(12, max(1, num("RETRIEVAL_K", "5"))),
        max_file_mb=max(1, num("MAX_FILE_MB", "25")),
        max_files=max(1, num("MAX_FILES", "10")),
        max_chunks=max(100, num("MAX_CHUNKS", "6000")),
        memory_turns=max(0, num("MEMORY_TURNS", "6")),
        temperature=min(1.0, max(0.0, num("TEMPERATURE", "0.2", float))),
        max_tokens=max(128, num("MAX_TOKENS", "900")),
    )
