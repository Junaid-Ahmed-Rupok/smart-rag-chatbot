"""
Session.py — single source of truth for st.session_state access.
"""

import logging
import uuid

import streamlit as st

from Config import SESSION_KEYS

log = logging.getLogger(__name__)

_SESSION_ID_KEY = "session_id"


# ── Lifecycle ──────────────────────────────────────────────────────────────────

def init() -> None:
    """
    Ensure all session keys exist with sane defaults.

    IMPORTANT: each default is created fresh, inline, right here — never
    read from a shared module-level list/dict/set. Modules are imported
    once per server process, so a module-level `[]` or `set()` used as a
    `st.session_state.setdefault()` default would be the SAME object
    handed to every visitor. The first append/add to it (e.g. one user
    sending a chat message) would then mutate that shared object, and
    every subsequent new/refreshed session would be initialised with
    that leftover data from a completely different session. Building
    the default inline in each setdefault() call guarantees a brand-new,
    unshared object every time.
    """
    st.session_state.setdefault(SESSION_KEYS["messages"], [])
    st.session_state.setdefault(SESSION_KEYS["processed_files"], set())
    st.session_state.setdefault(SESSION_KEYS["ollama_status"], False)


def get_session_id() -> str:
    """
    A short id unique to this browser session, used to scope each
    visitor's persisted vector store to their own directory so
    documents never leak between users or between separate chats
    on a shared deployment.
    """
    if _SESSION_ID_KEY not in st.session_state:
        st.session_state[_SESSION_ID_KEY] = uuid.uuid4().hex[:12]
    return st.session_state[_SESSION_ID_KEY]


# ── Messages ──────────────────────────────────────────────────────────────────

def messages() -> list:
    return st.session_state[SESSION_KEYS["messages"]]


def add_message(role: str, content: str, *, timestamp: str = "", sources: list | None = None) -> None:
    entry = {"role": role, "content": content, "timestamp": timestamp}
    if sources:
        entry["sources"] = sources
    messages().append(entry)


def turn_count() -> int:
    return sum(1 for m in messages() if m["role"] == "user")


def clear_chat() -> None:
    st.session_state[SESSION_KEYS["messages"]] = []
    log.info("Chat history cleared")


# ── Documents ─────────────────────────────────────────────────────────────────

def processed_files() -> set:
    return st.session_state[SESSION_KEYS["processed_files"]]


def mark_processed(filenames: list[str]) -> None:
    processed_files().update(filenames)


def has_documents() -> bool:
    return bool(processed_files())


# ── Ollama connection status ──────────────────────────────────────────────────

def ollama_status() -> bool:
    return st.session_state[SESSION_KEYS["ollama_status"]]


def set_ollama_status(is_running: bool) -> None:
    st.session_state[SESSION_KEYS["ollama_status"]] = is_running


# ── Full reset ────────────────────────────────────────────────────────────────

def reset_all() -> None:
    st.session_state[SESSION_KEYS["messages"]]        = []
    st.session_state[SESSION_KEYS["processed_files"]] = set()
    st.session_state.pop(SESSION_KEYS["rag_pipeline"], None)
    log.info("Session fully reset")
