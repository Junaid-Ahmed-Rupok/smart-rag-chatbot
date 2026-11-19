"""Sidebar: connection status, model, documents, settings, actions."""
from __future__ import annotations

import streamlit as st

from smartrag import embeddings
from smartrag.about import FOUNDER_MD
from smartrag.documents import IngestError, SUPPORTED_EXTENSIONS, load_file
from smartrag.export import chat_to_markdown
from smartrag.settings import APP_NAME, APP_VERSION

from . import state


def _chip(text: str, tone: str = "") -> str:
    return f'<span class="chip {tone}"><i></i>{text}</span>'


def _label(text: str) -> None:
    st.markdown(f'<div class="sidelabel">{text}</div>', unsafe_allow_html=True)


def index_loaded(name: str, data: bytes, progress=None) -> bool:
    """Read and index one file. Returns True on success; problems become friendly notices."""
    s = state.settings()
    kb = state.kb()
    if name not in kb.files and len(kb.files) >= s.max_files:
        state.notify("error", f"You can index up to {s.max_files} documents. Remove one first.")
        return False
    try:
        info = kb.add(load_file(name, data, s.max_file_bytes), progress)
        state.notify("success", f"Indexed **{info.name}**: {info.chunks} passages.")
        return True
    except IngestError as exc:
        state.notify("error", str(exc))
    except Exception:
        state.notify("error", f"{name} could not be indexed. Please try a different file.")
    return False


def _status_block() -> None:
    llm = state.llm()
    s = state.settings()
    if llm.available:
        chips = [_chip(f"{ {'groq': 'Groq', 'ollama': 'Ollama'}.get(llm.provider, llm.provider.title()) } connected", "ok")]
    else:
        chips = [_chip("Offline mode: no API key", "warn")]
    kb = state.kb()
    emb = embeddings.status()
    if kb.has_documents():
        chips.append(_chip("Hybrid search" if kb.semantic else "Keyword search", "ok" if kb.semantic else "warn"))
    else:
        chips.append(_chip({"ready": "Search model ready", "loading": "Search model loading",
                            "unavailable": "Keyword search only"}.get(emb, "Search model idle"),
                           "ok" if emb == "ready" else "warn" if emb in ("loading", "unavailable") else ""))
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def _model_picker() -> None:
    llm = state.llm()
    if not llm.available:
        with st.expander("Turn on AI answers"):
            st.markdown(
                "1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)\n"
                "2. Add it as `GROQ_API_KEY` (Streamlit **Secrets**, or a `.env` file)\n"
                "3. Restart the app"
            )
        return
    models = llm.list_models() or [state.settings().groq_model]
    default = state.settings().groq_model if llm.provider == "groq" else state.settings().ollama_model
    if "model" not in st.session_state or st.session_state["model"] not in models:
        st.session_state["model"] = default if default in models else models[0]
    _label("Model")
    st.selectbox("Model", models, key="model", label_visibility="collapsed")


def _documents() -> None:
    s = state.settings()
    kb = state.kb()
    _label("Documents")
    uploads = st.file_uploader(
        "Upload documents", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True, key=f"uploader_{st.session_state['uploader_n']}",
        label_visibility="collapsed", help=f"PDF, DOCX, TXT, MD or CSV. Up to {s.max_file_mb} MB each.",
    )
    fresh = [f for f in (uploads or []) if f.file_id not in st.session_state["seen_uploads"]]
    if fresh:
        with st.status("Reading your documents…", expanded=True) as box:
            bar = st.progress(0.0)
            for i, f in enumerate(fresh, start=1):
                bar.progress((i - 1) / len(fresh), text=f"{f.name} ({i}/{len(fresh)})")
                index_loaded(f.name, f.getvalue(),
                             lambda msg, frac, i=i: bar.progress(min(1.0, (i - 1 + frac) / len(fresh)), text=msg))
                st.session_state["seen_uploads"].add(f.file_id)
            bar.empty()
            box.update(label="Documents ready", state="complete", expanded=False)
        st.session_state["uploader_n"] += 1          # empty the upload box; the list below is the source of truth
        st.rerun()

    if not kb.has_documents():
        if st.button("Try a sample document", width="stretch", type="secondary", key="sample_btn"):
            index_loaded(state.SAMPLE_PATH.name, state.SAMPLE_PATH.read_bytes())
            st.rerun()
    for name, info in list(kb.files.items()):
        a, b = st.columns([6, 1], vertical_alignment="center")
        pages = f"{info.pages} pages · " if info.pages else ""
        a.markdown(f'<div class="doc">📄 {name}</div><div class="docmeta">{pages}{info.chunks} passages</div>',
                   unsafe_allow_html=True)
        if b.button("", icon=":material/close:", key=f"rm_{name}", help=f"Remove {name}"):
            kb.remove(name)
            st.rerun()


def _settings_panel() -> None:
    with st.expander("Settings"):
        st.toggle("Answer from documents only", key="documents_only",
                  help="On: the bot says so when your documents do not cover a question. "
                       "Off: it may fall back to general knowledge.")
        st.slider("Passages per answer", 2, 10, key="top_k",
                  help="More passages give the model more to read, but cost more of the free-tier quota.")


def _actions() -> None:
    kb = state.kb()
    msgs = st.session_state["messages"]
    _label("Session")
    c1, c2 = st.columns(2)
    if c1.button("Clear chat", width="stretch", disabled=not msgs):
        st.session_state["messages"] = []
        st.rerun()
    if c2.button("Remove all", width="stretch", disabled=not kb.has_documents(),
                 help="Remove every document from this session"):
        kb.clear()
        st.session_state["uploader_n"] += 1
        st.session_state["seen_uploads"] = set()
        st.rerun()
    if msgs:
        st.download_button("Download chat (.md)", chat_to_markdown(msgs, list(kb.files)),
                           file_name="chat.md", mime="text/markdown", width="stretch")


def render() -> None:
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(f"Version {APP_VERSION}")
        _status_block()
        _model_picker()
        _documents()
        _settings_panel()
        _actions()
        with st.expander("About the builder"):
            st.markdown(FOUNDER_MD)
        st.caption("Your documents stay in memory for this session only. Nothing is saved to disk.")
