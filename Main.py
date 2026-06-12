"""
app.py — Smart RAG Chatbot entry point.
Run: streamlit run app.py
"""

import logging
from datetime import datetime
from typing import Optional

import requests
import streamlit as st

from design.components import apply_professional_theme, ds
from rag import get_rag_pipeline, init_rag
from settings import APP_ICON, APP_NAME, APP_VERSION, SESSION_KEYS, cfg, bootstrap

log = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────

apply_professional_theme()
bootstrap(cfg)
init_rag()

# ── Session state ─────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    SESSION_KEYS["messages"]:        [],
    SESSION_KEYS["processed_files"]: set(),
    SESSION_KEYS["ollama_status"]:   False,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def msgs() -> list:
    return st.session_state[SESSION_KEYS["messages"]]


def docs() -> set:
    return st.session_state[SESSION_KEYS["processed_files"]]


# ── Ollama probe ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def probe_ollama(host: str) -> tuple[bool, list[str]]:
    """
    Poll Ollama once per minute.
    Returns (is_running, installed_model_names).
    """
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        names = [
            m.get("name", "unknown").split(":")[0]
            for m in r.json().get("models", [])
        ]
        return True, names if names else []
    except requests.RequestException:
        return False, []


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> Optional[str]:
    """
    Renders the sidebar. Returns the selected model name, or None if unavailable.
    """
    selected_model: Optional[str] = None

    with st.sidebar:
        # Brand
        st.markdown(
            f"""
            <div style="padding:1rem 0 .5rem;text-align:center">
                <div style="font-size:2.5rem">🤖</div>
                <div style="font-weight:700;font-size:1rem;color:var(--text-primary)">{APP_NAME}</div>
                <div style="font-size:.75rem;color:var(--text-muted)">v{APP_VERSION} &middot; local &middot; free</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ds.divider()

        # Model selection
        st.markdown(
            '<p style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:.06em;color:var(--text-muted);margin-bottom:.5rem">Model</p>',
            unsafe_allow_html=True,
        )

        ollama_ok, installed = probe_ollama(cfg.ollama_host)
        st.session_state[SESSION_KEYS["ollama_status"]] = ollama_ok

        if ollama_ok and installed:
            ds.status_badge("Ollama running", "success")
            st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)
            selected_model = st.selectbox(
                "Active model",
                installed,
                label_visibility="collapsed",
            )
        elif ollama_ok:
            ds.alert("No models installed", "Run `ollama pull mistral` in your terminal.", "warning")
        else:
            ds.status_badge("Ollama offline", "error")
            st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)
            with st.expander("How to start Ollama"):
                st.code("ollama serve\nollama pull mistral", language="bash")

        ds.divider()

        # Documents
        st.markdown(
            '<p style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:.06em;color:var(--text-muted);margin-bottom:.5rem">Documents</p>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        new_files = [f for f in (uploaded or []) if f.name not in docs()]

        if new_files:
            st.markdown("<div style='margin:.4rem 0'></div>", unsafe_allow_html=True)
            ds.status_badge(f"{len(new_files)} file(s) pending", "info")
            st.markdown("<div style='margin:.4rem 0'></div>", unsafe_allow_html=True)

            if st.button("Index documents", use_container_width=True, type="primary"):
                if not ollama_ok:
                    ds.toast("Start Ollama before indexing", "error")
                else:
                    try:
                        pipeline = get_rag_pipeline(selected_model)
                        with st.spinner(f"Indexing {len(new_files)} file(s)..."):
                            chunk_count = pipeline.process_documents(new_files)
                        for f in new_files:
                            docs().add(f.name)
                        ds.toast(f"Indexed {chunk_count:,} chunks", "success")
                        log.info("Indexed %d files -> %d chunks", len(new_files), chunk_count)
                        st.rerun()
                    except Exception as exc:
                        log.exception("Indexing failed")
                        ds.alert("Indexing failed", str(exc), "error")

        if docs():
            st.markdown("<div style='margin:.75rem 0 .25rem'></div>", unsafe_allow_html=True)
            for name in sorted(docs()):
                ds.source_card(name)

        ds.divider()

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            ds.metric_card("Docs", str(len(docs())), "📄")
        with col2:
            turns = sum(1 for m in msgs() if m["role"] == "user")
            ds.metric_card("Turns", str(turns), "💬")

        ds.divider()

        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat", use_container_width=True):
                st.session_state[SESSION_KEYS["messages"]] = []
                st.rerun()
        with col2:
            if st.button("Reset all", use_container_width=True):
                st.session_state[SESSION_KEYS["messages"]]        = []
                st.session_state[SESSION_KEYS["processed_files"]] = set()
                st.session_state.pop(SESSION_KEYS["rag_pipeline"], None)
                st.rerun()

    return selected_model


# ── Empty state ───────────────────────────────────────────────────────────────

def _render_empty_state(has_docs: bool) -> None:
    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
    icon    = "💬" if has_docs else "📂"
    heading = "Ask your first question" if has_docs else "No documents indexed yet"
    body    = (
        "Your documents are ready. Type a question below and I'll search them for an answer."
        if has_docs else
        "Upload PDF, DOCX, or TXT files in the sidebar, then click <strong>Index documents</strong>."
    )
    st.markdown(
        f"""
        <div style="text-align:center;padding:3rem 1rem;max-width:480px;margin:0 auto">
            <div style="font-size:3.5rem;margin-bottom:1rem">{icon}</div>
            <h2 style="color:var(--text-primary);font-size:1.25rem;font-weight:600;margin-bottom:.5rem">
                {heading}
            </h2>
            <p style="color:var(--text-muted);font-size:.9rem;line-height:1.6;margin:0">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Chat ──────────────────────────────────────────────────────────────────────

def _render_history() -> None:
    for msg in msgs():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                    for src in msg["sources"]:
                        ds.source_card(src)
            if ts := msg.get("timestamp"):
                st.caption(ts)


def _handle_prompt(prompt: str, model: str) -> None:
    ts = datetime.now().strftime("%I:%M %p")
    msgs().append({"role": "user", "content": prompt, "timestamp": ts})

    with st.chat_message("user"):
        st.markdown(prompt)

    answer: str = ""
    sources: list = []

    with st.chat_message("assistant"):
        try:
            pipeline   = get_rag_pipeline(model)
            docs_ready = bool(docs())

            with st.spinner(f"Thinking with {model}..."):
                if docs_ready:
                    result  = pipeline.ask(prompt)
                    answer  = result["answer"]
                    sources = result.get("sources", [])
                else:
                    pipeline.init_models()
                    answer = pipeline.llm.invoke(prompt)

            st.markdown(answer)

            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    for src in sources:
                        ds.source_card(src)

        except requests.ConnectionError:
            answer = "Lost connection to Ollama. Make sure `ollama serve` is still running."
            log.warning("Ollama connection lost during inference")
            ds.alert("Connection lost", answer, "error")

        except ValueError as exc:
            answer = f"Configuration error: {exc}"
            log.error("ValueError during inference: %s", exc)
            ds.alert("Configuration error", str(exc), "error")

        except Exception as exc:
            answer = "An unexpected error occurred. Check the terminal for details."
            log.exception("Unhandled inference error")
            ds.alert("Unexpected error", str(exc), "error")

        finally:
            msgs().append({
                "role":      "assistant",
                "content":   answer,
                "sources":   sources,
                "timestamp": datetime.now().strftime("%I:%M %p"),
            })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    model = _render_sidebar()

    # Hard gate: Ollama must be reachable
    if not st.session_state[SESSION_KEYS["ollama_status"]]:
        ds.header(APP_NAME, icon=APP_ICON)
        ds.alert(
            "Ollama is not running",
            "Open a terminal and run `ollama serve`, then refresh this page.",
            "error",
        )
        st.stop()

    # Soft gate: Ollama is up but no models are pulled
    if model is None:
        ds.header(APP_NAME, icon=APP_ICON)
        ds.alert("No models available", "Run: `ollama pull mistral`", "warning")
        st.stop()

    ds.header(APP_NAME, icon=APP_ICON)

    if msgs():
        _render_history()
    else:
        _render_empty_state(has_docs=bool(docs()))

    prompt = st.chat_input(
        placeholder="Ask about your documents..." if bool(docs()) else "Ask a general question...",
    )
    if prompt:
        _handle_prompt(prompt.strip(), model)


main()
