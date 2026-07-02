"""
Sidebar.py — branding, model selection, document upload, stats, and session controls.
"""

import logging
import requests
import streamlit as st

import Session
from design.components import ds
from Rag import get_rag_pipeline, delete_persisted_store, clear_chat_memory
from Config import APP_NAME, APP_VERSION, GROQ_AVAILABLE_MODELS, cfg

log = logging.getLogger(__name__)


# ── Provider probes ──────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _probe_groq(api_key: str) -> bool:
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def _probe_ollama(host: str) -> tuple[bool, list[str]]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        names = [
            m.get("name", "unknown").split(":")[0]
            for m in r.json().get("models", [])
        ]
        return True, names
    except requests.RequestException:
        return False, []


# ── Sections ──────────────────────────────────────────────────────────────────

def _render_brand() -> None:
    tagline = "local · free" if cfg.llm_provider == "ollama" else "powered by Groq"
    st.markdown(
        f"""
        <div style="padding:1rem 0 .5rem;text-align:center">
            <div style="font-size:2.5rem">🤖</div>
            <div style="font-weight:700;font-size:1rem;color:var(--text-primary)">{APP_NAME}</div>
            <div style="font-size:.75rem;color:var(--text-muted)">v{APP_VERSION} &middot; {tagline}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_label(text: str) -> None:
    st.markdown(
        f'<p style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:.06em;color:var(--text-muted);margin-bottom:.5rem">{text}</p>',
        unsafe_allow_html=True,
    )


def _render_model_section() -> tuple[str | None, bool]:
    _section_label("Model")

    if cfg.llm_provider == "groq":
        groq_ok = _probe_groq(cfg.groq_api_key)
        Session.set_ollama_status(groq_ok)

        if groq_ok:
            ds.status_badge("Groq connected", "success")
            st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)
            default_idx = (
                GROQ_AVAILABLE_MODELS.index(cfg.groq_model)
                if cfg.groq_model in GROQ_AVAILABLE_MODELS else 0
            )
            selected_model = st.selectbox(
                "Active model",
                GROQ_AVAILABLE_MODELS,
                index=default_idx,
                label_visibility="collapsed",
            )
        else:
            selected_model = None
            ds.status_badge("Groq unreachable", "error")
            st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)
            with st.expander("Troubleshooting"):
                st.write("Check your internet connection and that `GROQ_API_KEY` in `.env` is a valid key from console.groq.com/keys.")

        return selected_model, groq_ok

    # ── Ollama path ──
    ollama_ok, installed = _probe_ollama(cfg.ollama_host)
    Session.set_ollama_status(ollama_ok)

    selected_model: str | None = None

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

    return selected_model, ollama_ok


def _render_documents_section(selected_model: str | None, provider_ready: bool) -> None:
    _section_label("Documents")

    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    new_files = [f for f in (uploaded or []) if f.name not in Session.processed_files()]

    if new_files:
        st.markdown("<div style='margin:.4rem 0'></div>", unsafe_allow_html=True)
        ds.status_badge(f"{len(new_files)} file(s) pending", "info")
        st.markdown("<div style='margin:.4rem 0'></div>", unsafe_allow_html=True)

        if st.button("Index documents", use_container_width=True, type="primary"):
            if not provider_ready:
                msg = "Check your Groq connection" if cfg.llm_provider == "groq" else "Start Ollama"
                ds.toast(f"{msg} before indexing", "error")
            elif selected_model is None:
                ds.toast("No model selected", "error")
            else:
                try:
                    pipeline = get_rag_pipeline(selected_model)

                    progress_bar = st.progress(0, text="Starting…")

                    def _on_file_progress(started: int, total: int, filename: str) -> None:
                        # started = 1-based count of files begun so far.
                        # Shown as a fraction so the bar visibly advances
                        # with each file rather than jumping once at the end.
                        fraction = started / total
                        progress_bar.progress(
                            fraction,
                            text=f"Indexing {filename} ({started}/{total})",
                        )

                    chunk_count = pipeline.process_documents(
                        new_files, progress_callback=_on_file_progress
                    )

                    progress_bar.progress(1.0, text="Done")
                    progress_bar.empty()

                    Session.mark_processed([f.name for f in new_files])
                    ds.toast(f"Indexed {chunk_count:,} chunks", "success")
                    log.info("Indexed %d files -> %d chunks", len(new_files), chunk_count)
                    st.rerun()
                except Exception as exc:
                    log.exception("Indexing failed")
                    ds.alert("Indexing failed", str(exc), "error")

    if Session.has_documents():
        st.markdown("<div style='margin:.75rem 0 .25rem'></div>", unsafe_allow_html=True)
        for name in sorted(Session.processed_files()):
            ds.source_card(name)


def _render_stats() -> None:
    col1, col2 = st.columns(2)
    with col1:
        ds.metric_card("Docs", str(len(Session.processed_files())), "📄")
    with col2:
        ds.metric_card("Turns", str(Session.turn_count()), "💬")


def _render_actions() -> None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            Session.clear_chat()
            clear_chat_memory(Session.get_session_id())
            st.rerun()
    with col2:
        if st.button("Reset all", use_container_width=True):
            delete_persisted_store(Session.get_session_id())
            Session.reset_all()
            st.rerun()


# ── Entry point for Main.py ────────────────────────────────────────────────────

def render() -> str | None:
    with st.sidebar:
        _render_brand()
        ds.divider()

        selected_model, provider_ready = _render_model_section()
        ds.divider()

        _render_documents_section(selected_model, provider_ready)
        ds.divider()

        _render_stats()
        ds.divider()

        _render_actions()

    return selected_model
