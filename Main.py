"""
Main.py — Smart RAG Chatbot entry point.

Pure orchestration: bootstraps the app, enforces the active provider's
readiness gates, and wires the three owning modules together. No
session-state access, no HTML, and no business logic lives here — that
belongs to Session.py, Sidebar.py, and Chat.py respectively. Keeping this
file thin is what lets those three modules stay testable in isolation.

Run: streamlit run Main.py
"""

import logging

import streamlit as st

from design.components import apply_professional_theme, ds
from Rag import init_rag
from Config import APP_ICON, APP_NAME, bootstrap, cfg

import Session
import Sidebar
import Chat

logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    # Page config / theme must be the first Streamlit call in the script.
    apply_professional_theme()

    bootstrap(cfg)
    init_rag()
    Session.init()

    model = Sidebar.render()

    if cfg.llm_provider == "groq":
        # Groq just needs a valid key — already validated at startup in
        # Config.py, so by the time we're here this can only be "no model
        # selected yet" (e.g. Sidebar couldn't reach Groq this run).
        if model is None:
            ds.header(APP_NAME, icon=APP_ICON)
            ds.alert(
                "Can't reach Groq",
                "Check your internet connection and that GROQ_API_KEY in .env is valid.",
                "error",
            )
            st.stop()
    else:
        # Hard gate: Ollama must be reachable at all.
        if not Session.ollama_status():
            ds.header(APP_NAME, icon=APP_ICON)
            ds.alert(
                "Ollama is not running",
                "Open a terminal and run `ollama serve`, then refresh this page.",
                "error",
            )
            st.stop()

        # Soft gate: Ollama is up, but no models are pulled yet.
        if model is None:
            ds.header(APP_NAME, icon=APP_ICON)
            ds.alert("No models available", "Run: `ollama pull mistral`", "warning")
            st.stop()

    ds.header(APP_NAME, icon=APP_ICON)
    Chat.render(model)


main()
