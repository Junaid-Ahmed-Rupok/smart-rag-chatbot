"""
Session.py — single source of truth for st.session_state access.

Every read or write to session state for chat/document/connection state
goes through this module. Nothing outside Session.py should reference
SESSION_KEYS directly for these concerns — that's what made the same
defaults dict (and the same two accessor functions) end up duplicated
in both Main.py and Chat.py.

Scope note: the cached RAGPipeline instance (SESSION_KEYS["rag_pipeline"])
is intentionally NOT managed here. It's tightly coupled to RAGPipeline's
own lifecycle (model-change invalidation) and is owned by Rag.py's
init_rag()/get_rag_pipeline(). reset_all() below clears it via the same
key so a full reset still works end-to-end, but Rag.py remains the
authority on what that value means.
"""

import logging

import streamlit as st

from Config import SESSION_KEYS

log = logging.getLogger(__name__)

_DEFAULTS: dict = {
    SESSION_KEYS["messages"]:        [],
    SESSION_KEYS["processed_files"]: set(),
    SESSION_KEYS["ollama_status"]:   False,
}


# ── Lifecycle ───────────────────────────────────────────────────────────────────

def init() -> None:
    """Ensure all session keys exist with sane defaults. Idempotent — safe
    to call on every rerun, since setdefault is a no-op once a key exists."""
    for key, default in _DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ── Messages ──────────────────────────────────────────────────────────────────

def messages() -> list:
    """The full conversation history for this session."""
    return st.session_state[SESSION_KEYS["messages"]]


def add_message(role: str, content: str, *, timestamp: str = "", sources: list | None = None) -> None:
    """Appends a single turn to the conversation history."""
    entry = {"role": role, "content": content, "timestamp": timestamp}
    if sources:
        entry["sources"] = sources
    messages().append(entry)


def turn_count() -> int:
    """Number of user-authored turns, used for the 'Turns' stat in the sidebar."""
    return sum(1 for m in messages() if m["role"] == "user")


def clear_chat() -> None:
    """Wipes conversation history but keeps indexed documents intact."""
    st.session_state[SESSION_KEYS["messages"]] = []
    log.info("Chat history cleared")


# ── Documents ─────────────────────────────────────────────────────────────────

def processed_files() -> set:
    """Filenames currently indexed in the vector store for this session."""
    return st.session_state[SESSION_KEYS["processed_files"]]


def mark_processed(filenames: list[str]) -> None:
    """Records a batch of filenames as successfully indexed."""
    processed_files().update(filenames)


def has_documents() -> bool:
    return bool(processed_files())


# ── Ollama connection status ────────────────────────────────────────────────────

def ollama_status() -> bool:
    return st.session_state[SESSION_KEYS["ollama_status"]]


def set_ollama_status(is_running: bool) -> None:
    st.session_state[SESSION_KEYS["ollama_status"]] = is_running


# ── Full reset ────────────────────────────────────────────────────────────────

def reset_all() -> None:
    """Wipes chat, indexed documents, and the cached RAG pipeline. The next
    question will rebuild everything from a clean slate."""
    st.session_state[SESSION_KEYS["messages"]]        = []
    st.session_state[SESSION_KEYS["processed_files"]] = set()
    st.session_state.pop(SESSION_KEYS["rag_pipeline"], None)
    log.info("Session fully reset")
