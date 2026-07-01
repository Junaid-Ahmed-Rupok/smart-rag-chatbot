"""
Main.py — bootstrap, provider gates, module wiring.

Owns page setup and orchestration only. All session-state access goes
through Session.py, all sidebar UI through Sidebar.py, and all chat
rendering/inference through Chat.py.
"""

import streamlit as st

from Config import APP_ICON, APP_NAME, APP_VERSION, bootstrap, cfg
from design.components import apply_professional_theme
import Session
import Sidebar
import Chat
from Rag import cleanup_stale_sessions, init_rag, touch_session_activity


def main() -> None:
    # 1. Page config — must be the very first Streamlit command, called once.
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Initialize session state (creates all default keys)
    Session.init()

    # 3. Bootstrap directories
    bootstrap(cfg)

    # 4. Janitor: purge any OTHER session's documents that have been
    #    inactive past the TTL — this is what makes uploaded files
    #    vanish once a chat is actually over.
    cleanup_stale_sessions()

    # 5. Heartbeat: mark THIS session as still active so its own
    #    documents survive while the chat is ongoing.
    touch_session_activity(Session.get_session_id())

    # 6. Apply custom theme (CSS only — page config already set above)
    apply_professional_theme()

    # 7. Initialize RAG session slot
    init_rag()

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(f"# {APP_ICON} {APP_NAME}")
    st.caption(f"v{APP_VERSION} · powered by Groq")

    # ── Sidebar (model selection, uploads, stats, controls) ────────────────
    selected_model = Sidebar.render()

    # ── Chat surface (history + input, handles a full turn) ────────────────
    model = selected_model or (cfg.groq_model if cfg.llm_provider == "groq" else cfg.default_model)
    Chat.render(model)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
