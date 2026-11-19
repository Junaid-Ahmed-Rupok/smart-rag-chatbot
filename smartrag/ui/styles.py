"""A small stylesheet on top of the native Streamlit dark theme (see .streamlit/config.toml)."""
import streamlit as st

CSS = """
<style>
.block-container { max-width: 880px; padding-top: 2.2rem; padding-bottom: 6rem; }
footer, [data-testid="stDecoration"] { display: none !important; }

h1 { letter-spacing: -0.02em; margin-bottom: 0.1rem; }
.tagline { color: #94A3B8; margin: 0 0 1.1rem 0; font-size: 1.02rem; }

/* status chips */
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 500; border: 1px solid #243049; background: #0F1A30; color: #CBD5E1; }
.chip i { width: 7px; height: 7px; border-radius: 50%; display: inline-block; background: #64748B; }
.chip.ok i { background: #34D399; } .chip.warn i { background: #FBBF24; } .chip.bad i { background: #F87171; }

/* welcome cards */
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1.2rem 0 1.4rem; }
.card { background: #0F1A30; border: 1px solid #1E2B45; border-radius: 12px; padding: 16px 16px 14px; }
.card b { display: block; margin: 6px 0 4px; font-size: 0.98rem; }
.card span { color: #94A3B8; font-size: 0.86rem; line-height: 1.5; }
.card .ico { font-size: 1.4rem; }
@media (max-width: 700px) { .cards { grid-template-columns: 1fr; } }

/* chat */
[data-testid="stChatMessage"] { background: transparent; padding: 0.4rem 0; }
[data-testid="stChatMessage"] p { line-height: 1.65; }
.meta { color: #64748B; font-size: 0.76rem; margin-top: 2px; }

/* source cards */
.src { border: 1px solid #1E2B45; background: #0F1A30; border-radius: 10px; padding: 10px 12px; margin: 6px 0; }
.src .h { font-size: 0.82rem; font-weight: 600; color: #93C5FD; margin-bottom: 4px; }
.src .t { font-size: 0.85rem; color: #CBD5E1; line-height: 1.55; border-left: 2px solid #2A3A5C; padding-left: 10px; }
.src mark { background: rgba(59,130,246,0.28); color: #F1F5F9; border-radius: 3px; padding: 0 2px; }
.src.dim { opacity: 0.8; }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid #1B2740; }
.sidelabel { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
             color: #64748B; margin: 0.9rem 0 0.35rem; }
.doc { font-size: 0.88rem; font-weight: 500; line-height: 1.3; word-break: break-word; }
.docmeta { font-size: 0.76rem; color: #64748B; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
