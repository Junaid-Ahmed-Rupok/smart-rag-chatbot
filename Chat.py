"""
Chat.py — conversation rendering and prompt handling.

Owns everything related to displaying chat history and running a single
turn of the conversation through the RAG pipeline. Knows nothing about
the sidebar or app bootstrapping — Main.py is responsible for wiring
this module into the page.
"""

import logging
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from groq import APIConnectionError as GroqConnectionError
from groq import AuthenticationError as GroqAuthError
from groq import RateLimitError as GroqRateLimitError

from design.components import ds
from Rag import get_rag_pipeline
from Config import SESSION_KEYS

log = logging.getLogger(__name__)


# ── Session accessors ──────────────────────────────────────────────────────────

def msgs() -> list:
    """The current session's message history (list of role/content dicts)."""
    return st.session_state[SESSION_KEYS["messages"]]


def docs() -> set:
    """Filenames currently indexed in the vector store for this session."""
    return st.session_state[SESSION_KEYS["processed_files"]]


# ── Empty state ─────────────────────────────────────────────────────────────────

def render_empty_state(has_docs: bool) -> None:
    """Shown before the first message is sent."""
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


# ── History ───────────────────────────────────────────────────────────────────

def render_history() -> None:
    """Replays every message in the session as a chat bubble."""
    for msg in msgs():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                    for src in msg["sources"]:
                        ds.source_card(src)
            if ts := msg.get("timestamp"):
                st.caption(ts)


# ── Turn handling ─────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%I:%M %p")


def handle_prompt(prompt: str, model: str) -> None:
    """
    Runs one conversational turn: appends the user message, queries the
    RAG pipeline (or the bare LLM if no documents are indexed), renders
    the response, and appends the assistant message.

    Never raises — all failure modes are caught, surfaced via the design
    system's alert component, and recorded as the assistant's reply so
    the conversation log stays consistent with what the user saw.
    """
    prompt = prompt.strip()
    if not prompt:
        return

    msgs().append({"role": "user", "content": prompt, "timestamp": _timestamp()})

    with st.chat_message("user"):
        st.markdown(prompt)

    answer: str = ""
    sources: list[str] = []

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
                    answer = pipeline.ask_direct(prompt)

            st.markdown(answer)

            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    for src in sources:
                        ds.source_card(src)

        except GroqAuthError:
            answer = "Invalid or missing Groq API key. Check GROQ_API_KEY in your .env file."
            log.error("Groq authentication failed")
            ds.alert("API key error", answer, "error")

        except GroqRateLimitError:
            answer = "Groq rate limit reached. Wait a moment and try again."
            log.warning("Groq rate limit hit")
            ds.alert("Rate limited", answer, "error")

        except GroqConnectionError:
            answer = "Couldn't reach Groq's API. Check your internet connection."
            log.warning("Groq connection error")
            ds.alert("Connection lost", answer, "error")

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
                "timestamp": _timestamp(),
            })


# ── Entry point for Main.py ────────────────────────────────────────────────────

def render(model: str) -> None:
    """
    Renders the full chat surface: history (or empty state) plus the
    input box. Call once per script run, after the sidebar and gates
    have already been handled by Main.py.
    """
    if msgs():
        render_history()
    else:
        render_empty_state(has_docs=bool(docs()))

    prompt: Optional[str] = st.chat_input(
        placeholder="Ask about your documents..." if bool(docs()) else "Ask a general question...",
    )
    if prompt:
        handle_prompt(prompt, model)
