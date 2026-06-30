"""
Config.py — single source of truth for all app configuration.
Import: from Config import cfg, AVAILABLE_MODELS, SUPPORTED_EXTENSIONS
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv(override=False)


# ── Constants (never change at runtime) ──────────────────────────────────────

APP_NAME        = "Smart RAG Chatbot"
APP_ICON        = "🤖"
APP_VERSION     = "2.0.0"

AVAILABLE_MODELS: List[str] = [
    "mistral",    # 4.0 GB — recommended
    "llama3",     # 4.6 GB — best quality
    "phi3",       # 2.3 GB — lightweight
    "gemma:2b",   # 1.6 GB — fastest
    "llama2",     # 3.8 GB — stable
    "codellama",  # 3.8 GB — code-focused
]

GROQ_AVAILABLE_MODELS: List[str] = [
    "llama-3.3-70b-versatile",   # best quality
    "llama-3.1-8b-instant",      # fastest
    "mixtral-8x7b-32768",        # long context
]

SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt"]

SESSION_KEYS = {
    "messages":        "chat_messages",
    "rag_pipeline":    "rag_pipeline_instance",
    "processed_files": "processed_documents_set",
    "vector_store":    "vector_store_instance",
    "uploaded_files":  "uploaded_files_list",
    "chain":           "conversation_chain",
    "ollama_status":   "ollama_connection_status",
}


# ── Config dataclass ──────────────────────────────────────────────────────────

def _bool(key: str, default: str = "False") -> bool:
    return os.getenv(key, default).lower() == "true"

def _int(key: str, default: str) -> int:
    return int(os.getenv(key, default))

def _float(key: str, default: str) -> float:
    return float(os.getenv(key, default))

def _path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default))


@dataclass(frozen=True)
class Settings:
    # App
    debug: bool = field(default_factory=lambda: _bool("DEBUG"))

    # Ollama
    ollama_host:    str   = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    default_model:  str   = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "mistral"))
    embedding_model: str  = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
    temperature:    float = field(default_factory=lambda: _float("TEMPERATURE", "0.2"))
    top_p:          float = field(default_factory=lambda: _float("TOP_P", "0.95"))
    repeat_penalty: float = field(default_factory=lambda: _float("REPEAT_PENALTY", "1.1"))

    # LLM provider — "ollama" (local, free, private) or "groq" (hosted, no
    # install required for end users). Embeddings stay 100% local either way
    # (sentence-transformers), so no second API key is ever required.
    llm_provider:   str            = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    groq_api_key:   str | None     = field(default_factory=lambda: os.getenv("GROQ_API_KEY") or None)
    groq_model:     str            = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    local_embedding_model: str     = field(default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    # RAG
    chunk_size:      int   = field(default_factory=lambda: _int("CHUNK_SIZE", "1000"))
    chunk_overlap:   int   = field(default_factory=lambda: _int("CHUNK_OVERLAP", "200"))
    retrieval_k:     int   = field(default_factory=lambda: _int("RETRIEVAL_K", "4"))
    retrieval_fetch_k: int = field(default_factory=lambda: _int("RETRIEVAL_FETCH_K", "8"))
    search_type:     str   = field(default_factory=lambda: os.getenv("SEARCH_TYPE", "mmr"))
    mmr_lambda:      float = field(default_factory=lambda: _float("MMR_LAMBDA", "0.5"))
    memory_length:   int   = field(default_factory=lambda: _int("CONVERSATION_MEMORY_LENGTH", "10"))

    # Files
    max_upload_mb:      int  = field(default_factory=lambda: _int("MAX_UPLOAD_SIZE_MB", "200"))
    vector_store_type:  str  = field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss"))
    vector_store_path:  Path = field(default_factory=lambda: _path("VECTOR_STORE_PATH", "./data/vector_store"))
    data_path:          Path = field(default_factory=lambda: Path("./data"))

    # Derived
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def chunk_separators(self) -> List[str]:
        return ["\n\n", "\n", " ", ""]

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"TEMPERATURE must be 0–2, got {self.temperature}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.retrieval_k < 1:
            raise ValueError("RETRIEVAL_K must be >= 1")
        if self.retrieval_fetch_k < self.retrieval_k:
            raise ValueError("RETRIEVAL_FETCH_K must be >= RETRIEVAL_K")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"MMR_LAMBDA must be 0–1, got {self.mmr_lambda}")
        if self.search_type not in ("mmr", "similarity"):
            raise ValueError(f"SEARCH_TYPE must be 'mmr' or 'similarity', got {self.search_type!r}")
        if self.llm_provider not in ("ollama", "groq"):
            raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'groq', got {self.llm_provider!r}")
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY to be set. "
                "Get a free key at https://console.groq.com/keys"
            )

    def validate(self) -> dict[str, bool]:
        """Runtime checks (writable paths etc.)"""
        return {
            "chunk_size_valid":       100 <= self.chunk_size <= 5000,
            "chunk_overlap_valid":    0 <= self.chunk_overlap < self.chunk_size,
            "retrieval_k_valid":      1 <= self.retrieval_k <= 20,
            "temperature_valid":      0 <= self.temperature <= 2,
            "data_path_writable":     self.data_path.exists() and os.access(self.data_path, os.W_OK),
            "vector_store_writable":  self.vector_store_path.exists() and os.access(self.vector_store_path, os.W_OK),
        }

    def is_ready(self) -> bool:
        v = self.validate()
        return all(v.values())

    def summary(self) -> dict:
        return {
            "llm_provider": self.llm_provider,
            "model":        self.groq_model if self.llm_provider == "groq" else self.default_model,
            "groq_key":     "set (hidden)" if self.groq_api_key else "not set",
            "host":         self.ollama_host,
            "temperature":  self.temperature,
            "chunk_size":   self.chunk_size,
            "retrieval_k":  self.retrieval_k,
            "search_type":  self.search_type,
            "memory":       self.memory_length,
            "upload_limit": f"{self.max_upload_mb} MB",
        }


# ── Bootstrap (create dirs) — call explicitly, not at import ─────────────────

def bootstrap(cfg: Settings) -> None:
    """Create required directories. Call once at app startup."""
    for p in [
        cfg.data_path,
        cfg.data_path / "raw",
        cfg.data_path / "processed",
        cfg.vector_store_path,
        Path("./logs"),
    ]:
        p.mkdir(parents=True, exist_ok=True)

    if cfg.debug:
        import pprint
        print(f"\n{APP_NAME} v{APP_VERSION} — debug mode")
        pprint.pprint(cfg.summary())


# ── Singleton ─────────────────────────────────────────────────────────────────

cfg = Settings()
