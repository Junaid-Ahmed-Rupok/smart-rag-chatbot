"""Per-visitor state. Everything lives in memory for the session; nothing is written to disk."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from smartrag import embeddings
from smartrag.engine import Engine
from smartrag.index import KnowledgeBase
from smartrag.llm import make_llm
from smartrag.settings import Settings, load_settings

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "samples" / "how-rag-works.md"


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    return load_settings()


def init() -> None:
    s = settings()
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("documents_only", True)
    ss.setdefault("top_k", s.top_k)
    ss.setdefault("uploader_n", 0)
    ss.setdefault("seen_uploads", set())
    ss.setdefault("notices", [])
    ss.setdefault("pending", None)
    if "kb" not in ss:
        ss["kb"] = KnowledgeBase(
            embedder_factory=lambda: embeddings.get_embedder(s.embedding_model),
            chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, max_chunks=s.max_chunks,
        )


def kb() -> KnowledgeBase:
    return st.session_state["kb"]


def llm():
    return make_llm(settings())


def engine() -> Engine:
    return Engine(kb(), llm(), settings())


def notify(level: str, text: str) -> None:
    st.session_state["notices"].append((level, text))


def flush_notices() -> None:
    items, st.session_state["notices"] = st.session_state["notices"], []
    for level, text in items:
        {"error": st.error, "warning": st.warning, "success": st.success}.get(level, st.info)(text)
