"""The main chat surface: welcome screen, history, streaming answers and source cards."""
from __future__ import annotations

import html
import itertools
import re
from dataclasses import asdict

import streamlit as st

from smartrag.bm25 import tokenize
from smartrag.settings import APP_NAME, TAGLINE

from . import state

SUGGESTIONS = ["Summarize the documents", "What are the key points?", "List important names, dates or numbers"]
SAMPLE_QUESTIONS = ["What is chunk overlap?", "Why use hybrid search?", "Summarize the document"]


def _highlight(text: str, query: str) -> str:
    safe = html.escape(" ".join(text.split()))
    words = sorted({w for w in re.findall(r"\w{3,}", query.lower()) if tokenize(w)}, key=len, reverse=True)
    if words:
        safe = re.sub(r"(?<!\w)((?:" + "|".join(map(re.escape, words)) + r")\w*)", r"<mark>\1</mark>", safe, flags=re.I)
    return safe if len(safe) < 1100 else safe[:1100] + "…"


def render_sources(citations: list[dict], query: str = "") -> None:
    used = [c for c in citations if c["used"]]
    shown, title = (used, "Sources") if used else (citations, "Passages I looked at (not cited)")
    if not shown:
        return
    with st.expander(f"{title} ({len(shown)})", expanded=False):
        for c in shown:
            st.markdown(
                f'<div class="src {"" if c["used"] else "dim"}"><div class="h">[{c["n"]}] {html.escape(c["label"])}</div>'
                f'<div class="t">{_highlight(c["text"], query)}</div></div>',
                unsafe_allow_html=True,
            )


def _meta(m: dict) -> str:
    bits = []
    if m.get("model"):
        bits.append(m["model"])
    if m.get("seconds"):
        bits.append(f"{m['seconds']}s")
    n = len(m.get("citations", []))
    if m.get("kind") in ("answer", "extractive") and n:
        bits.append(f"{n} passages searched · {m.get('mode', 'hybrid')} search")
    if m.get("kind") == "refusal":
        bits.append("no matching passages")
    return " · ".join(bits)


def render_history() -> None:
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("citations", []), m.get("standalone", ""))
                if meta := _meta(m):
                    st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)


def render_welcome() -> None:
    st.markdown(
        """
        <div class="cards">
          <div class="card"><div class="ico">📄</div><b>Bring your documents</b>
            <span>PDF, Word, Markdown, text or CSV. Page numbers and sections are kept.</span></div>
          <div class="card"><div class="ico">🔎</div><b>Hybrid search</b>
            <span>Keyword and semantic search are combined to find the right passages.</span></div>
          <div class="card"><div class="ico">✅</div><b>Cited, grounded answers</b>
            <span>Every claim points to its source, and the bot says so when it cannot find an answer.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload documents in the sidebar, or click **Try a sample document** to see it work in seconds.",
            icon="👈")


def _suggestions() -> None:
    kb = state.kb()
    options = SAMPLE_QUESTIONS if "how-rag-works.md" in kb.files and len(kb.files) == 1 else SUGGESTIONS
    cols = st.columns(len(options))
    for col, q in zip(cols, options):
        if col.button(q, key=f"sug_{q}", width="stretch"):
            st.session_state["pending"] = q
            st.rerun()


def _stream_into_chat(turn) -> None:
    gen = turn.stream()
    with st.spinner("Searching your documents…" if state.kb().has_documents() else "Thinking…"):
        first = next(gen, None)
    if first is not None:
        st.write_stream(itertools.chain([first], gen))


def handle_prompt(prompt: str, intro=None) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    if intro is not None:
        intro.empty()                      # remove the welcome / suggestion block as soon as the chat starts
    ss = st.session_state
    history = list(ss["messages"])
    ss["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    turn = state.engine().ask(
        prompt, history, documents_only=ss["documents_only"],
        model=ss.get("model", ""), top_k=ss["top_k"],
    )
    with st.chat_message("assistant", avatar="🤖"):
        _stream_into_chat(turn)
        answer = turn.result()
        entry = {
            "role": "assistant", "content": answer.text, "kind": answer.kind,
            "citations": [asdict(c) for c in answer.citations], "model": answer.model,
            "seconds": answer.seconds, "mode": answer.mode, "standalone": answer.standalone,
        }
        ss["messages"].append(entry)
        render_sources(entry["citations"], answer.standalone)
        if meta := _meta(entry):
            st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)


def render() -> None:
    kb = state.kb()
    st.markdown(f"# {APP_NAME}")
    st.markdown(f'<p class="tagline">{TAGLINE}</p>', unsafe_allow_html=True)
    state.flush_notices()

    intro = st.empty()
    if st.session_state["messages"]:
        render_history()
    elif kb.has_documents():
        with intro.container():
            st.markdown(f"**{len(kb.files)} document{'s' if len(kb.files) != 1 else ''} ready** · "
                        f"{len(kb):,} passages. Ask anything, or start with one of these:")
            _suggestions()
    else:
        with intro.container():
            render_welcome()

    typed = st.chat_input("Ask about your documents…" if kb.has_documents() else "Ask a question, or upload documents first…")
    prompt = typed or st.session_state.pop("pending", None)
    if prompt:
        handle_prompt(prompt, intro)
