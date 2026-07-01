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
import uuid

import streamlit as st

from Config import SESSION_KEYS, cfg

log = logging.getLogger(__name__)

_DEFAULTS: dict = {
    SESSION_KEYS["messages"]:        [],
    SESSION_KEYS["processed_files"]: set(),
    SESSION_KEYS["ollama_status"]:   False,
    SESSION_KEYS["session_id"]:      None,
}

# Fixed id used only when cfg.local_demo_mode is True (see Config.py).
# Never used on a shared deployment — see the warning on that flag.
_LOCAL_DEMO_SESSION_ID = "local-demo"


# ── Lifecycle ───────────────────────────────────────────────────────────────────

def init() -> None:
    """Ensure all session keys exist with sane defaults. Idempotent — safe
    to call on every rerun, since setdefault is a no-op once a key exists."""
    for key, default in _DEFAULTS.items():
        st.session_state.setdefault(key, default)

    if st.session_state[SESSION_KEYS["session_id"]] is None:
        if cfg.local_demo_mode:
            # Fixed id — every run on this laptop finds the same on-disk
            # index, no URL/bookmark required. Only safe because this
            # mode is opt-in and must stay off for shared deployments.
            sid = _LOCAL_DEMO_SESSION_ID
        else:
            # Recover the session id from the URL if present (survives a
            # full browser refresh, unlike session_state) — otherwise
            # mint a new one and stash it in the URL so future refreshes
            # find it too.
            sid = st.query_params.get("sid")
            if not sid:
                sid = uuid.uuid4().hex
                st.query_params["sid"] = sid

        st.session_state[SESSION_KEYS["session_id"]] = sid
        log.debug("session_id resolved: %s (local_demo_mode=%s)", sid, cfg.local_demo_mode)


# ── Session identity ─────────────────────────────────────────────────────────

def session_id() -> str:
    """Stable per-session identifier. Used to namespace persisted
    (on-disk) vector stores so one user's documents are never visible to
    another user on a shared deployment — unless local_demo_mode is on."""
    return st.session_state[SESSION_KEYS["session_id"]]


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
    question will rebuild everything from a clean slate.

    Note: this does NOT delete the persisted on-disk vector store — that's
    handled by Rag.delete_persisted_store(), called separately from
    Sidebar.py, to avoid a circular import between Session.py and Rag.py."""
    st.session_state[SESSION_KEYS["messages"]]        = []
    st.session_state[SESSION_KEYS["processed_files"]] = set()
    st.session_state.pop(SESSION_KEYS["rag_pipeline"], None)
    log.info("Session fully reset")
