"""
Professional Component Library
Enhanced: Dark mode, animations, new components
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

# ── Load external CSS ──────────────────────────────────────────────────────

def load_css():
    """Load the external style.css file."""
    css_path = Path(__file__).parent.parent / "style.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── CSS ──────────────────────────────────────────────────────────────────────

THEME_CSS = """
<style>
/* ── Root tokens (light mode default) ── */
:root {
    --blue-500: #1E88E5;
    --blue-700: #1565C0;
    --blue-100: #DBEAFE;
    --blue-900: #1E3A5F;

    --green-100: #D1FAE5;
    --green-800: #065F46;
    --yellow-100: #FEF3C7;
    --yellow-800: #92400E;
    --red-100:  #FEE2E2;
    --red-800:  #991B1B;

    --surface:      #FFFFFF;
    --surface-alt:  #F8FAFC;
    --border:       #E2E8F0;
    --text-primary: #0F172A;
    --text-muted:   #64748B;
    --text-hint:    #94A3B8;
    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    14px;
    --radius-pill:  9999px;
    --shadow-sm:    0 1px 3px rgba(0,0,0,.08);
    --shadow-md:    0 4px 12px rgba(0,0,0,.10);
    --ease:         cubic-bezier(.4,0,.2,1);
    --speed:        220ms;
}

/* ── Dark mode detection via Streamlit theme attribute ── */
[data-theme="dark"] {
    --blue-500:     #60A5FA;
    --blue-700:     #3B82F6;
    --blue-100:     #1E3A5F;
    --blue-900:     #BFDBFE;

    --green-100:    #064E3B;
    --green-800:    #6EE7B7;
    --yellow-100:   #451A03;
    --yellow-800:   #FDE68A;
    --red-100:      #450A0A;
    --red-800:      #FCA5A5;

    --surface:      #1E293B;
    --surface-alt:  #0F172A;
    --border:       #334155;
    --text-primary: #F1F5F9;
    --text-muted:   #94A3B8;
    --text-hint:    #64748B;
    --shadow-sm:    0 1px 3px rgba(0,0,0,.4);
    --shadow-md:    0 4px 12px rgba(0,0,0,.4);
}

/* Also support prefers-color-scheme for browsers without data attribute */
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --blue-500:     #60A5FA;
        --blue-700:     #3B82F6;
        --blue-100:     #1E3A5F;
        --blue-900:     #BFDBFE;

        --green-100:    #064E3B;
        --green-800:    #6EE7B7;
        --yellow-100:   #451A03;
        --yellow-800:   #FDE68A;
        --red-100:      #450A0A;
        --red-800:      #FCA5A5;

        --surface:      #1E293B;
        --surface-alt:  #0F172A;
        --border:       #334155;
        --text-primary: #F1F5F9;
        --text-muted:   #94A3B8;
        --text-hint:    #64748B;
        --shadow-sm:    0 1px 3px rgba(0,0,0,.4);
        --shadow-md:    0 4px 12px rgba(0,0,0,.4);
    }
}

/* ── Animations ── */
@keyframes fade-up {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes progress-fill {
    from { width: 0%; }
    to   { width: var(--pct); }
}
@keyframes skeleton-shimmer {
    from { background-position: -400px 0; }
    to   { background-position:  400px 0; }
}

/* ── Cards ── */
.ds-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    animation: fade-up var(--speed) var(--ease) both;
    transition: box-shadow var(--speed) var(--ease),
                transform   var(--speed) var(--ease);
    color: var(--text-primary);
}
.ds-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* ── Badge ── */
.ds-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: .25rem .65rem;
    border-radius: var(--radius-pill);
    font-size: .75rem;
    font-weight: 600;
    line-height: 1;
}

/* ── Progress bar ── */
.ds-progress-wrap {
    background: var(--border);
    border-radius: var(--radius-pill);
    overflow: hidden;
    height: 8px;
}
.ds-progress-fill {
    height: 100%;
    border-radius: var(--radius-pill);
    animation: progress-fill .8s var(--ease) both;
}

/* ── Table ── */
.ds-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .875rem;
    color: var(--text-primary);
}
.ds-table th {
    text-align: left;
    padding: .6rem 1rem;
    font-weight: 600;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .04em;
    color: var(--text-muted);
    border-bottom: 2px solid var(--border);
}
.ds-table td {
    padding: .75rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}
.ds-table tr:last-child td { border-bottom: none; }
.ds-table tr:hover td {
    background: var(--surface-alt);
    transition: background var(--speed) var(--ease);
}

/* ── Alert ── */
.ds-alert {
    display: flex;
    align-items: flex-start;
    gap: .75rem;
    padding: .9rem 1rem;
    border-radius: var(--radius-md);
    border-left: 3px solid;
    animation: fade-up var(--speed) var(--ease) both;
    color: var(--text-primary);
}
.ds-alert-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: .05rem; }
.ds-alert-title { font-weight: 600; font-size: .875rem; }
.ds-alert-body  { font-size: .8125rem; opacity: .85; margin-top: .2rem; }

/* ── Skeleton loader ── */
.ds-skeleton {
    border-radius: var(--radius-sm);
    background: linear-gradient(
        90deg,
        var(--surface-alt) 25%,
        var(--border) 50%,
        var(--surface-alt) 75%
    );
    background-size: 800px 100%;
    animation: skeleton-shimmer 1.6s infinite linear;
}
</style>
"""


# ── Design system class ───────────────────────────────────────────────────────

class DesignSystem:
    """Enterprise-grade design system — dark-mode aware, animated."""

    # ── Layout ──────────────────────────────────────────────────────────────

    @staticmethod
    def inject_theme():
        """Inject shared CSS (call once at app startup)."""
        # Load external style.css first (for chrome styling)
        load_css()
        # Then inject component CSS
        st.markdown(THEME_CSS, unsafe_allow_html=True)

    @staticmethod
    def header(
        title: str,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
    ):
        icon_html = f'<span style="font-size:2rem;margin-right:.5rem">{icon}</span>' if icon else ""
        sub_html = (
            f'<p style="color:var(--text-muted);margin:.4rem 0 0">{subtitle}</p>'
            if subtitle else ""
        )
        st.markdown(
            f"""
            <div style="margin-bottom:2rem;animation:fade-up 300ms cubic-bezier(.4,0,.2,1) both">
                <div style="display:flex;align-items:center">
                    {icon_html}
                    <h1 style="color:var(--blue-500);margin:0;font-size:2rem;font-weight:700">{title}</h1>
                </div>
                {sub_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def divider():
        st.markdown(
            '<hr style="margin:1.5rem 0;border:none;border-top:1px solid var(--border)">',
            unsafe_allow_html=True,
        )

    # ── Metric card ─────────────────────────────────────────────────────────

    @staticmethod
    def metric_card(
        label: str,
        value: str,
        icon: str,
        delta: Optional[str] = None,
        delta_positive: bool = True,
    ):
        delta_color = "var(--green-800)" if delta_positive else "var(--red-800)"
        delta_html = (
            f'<div style="font-size:.75rem;color:{delta_color};margin-top:.2rem">{delta}</div>'
            if delta else ""
        )
        st.markdown(
            f"""
            <div class="ds-card" style="text-align:center">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-size:.8125rem;color:var(--text-muted);margin-top:.4rem">{label}</div>
                <div style="font-size:1.75rem;font-weight:700;color:var(--blue-500);margin-top:.2rem">{value}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Status badge ─────────────────────────────────────────────────────────

    @staticmethod
    def status_badge(status: str, type: str = "info"):
        palettes = {
            "success": ("var(--green-100)", "var(--green-800)", "●"),
            "warning": ("var(--yellow-100)", "var(--yellow-800)", "●"),
            "error":   ("var(--red-100)",    "var(--red-800)",   "●"),
            "info":    ("var(--blue-100)",   "var(--blue-900)",  "●"),
        }
        bg, fg, dot = palettes.get(type, palettes["info"])
        st.markdown(
            f"""
            <span class="ds-badge" style="background:{bg};color:{fg}">
                <span style="font-size:.5rem">{dot}</span>{status}
            </span>
            """,
            unsafe_allow_html=True,
        )

    # ── Progress bar ─────────────────────────────────────────────────────────

    @staticmethod
    def progress_bar(
        label: str,
        value: float,          # 0.0 – 1.0
        color: str = "var(--blue-500)",
        show_pct: bool = True,
    ):
        pct = max(0.0, min(1.0, value)) * 100
        label_right = f'<span style="color:var(--text-muted);font-size:.8125rem">{pct:.0f}%</span>' if show_pct else ""
        st.markdown(
            f"""
            <div style="margin:.5rem 0">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem">
                    <span style="font-size:.875rem;color:var(--text-primary)">{label}</span>
                    {label_right}
                </div>
                <div class="ds-progress-wrap">
                    <div class="ds-progress-fill"
                         style="width:{pct:.1f}%;background:{color};--pct:{pct:.1f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Alert / inline notification ──────────────────────────────────────────

    @staticmethod
    def alert(
        title: str,
        body: Optional[str] = None,
        type: str = "info",
    ):
        palettes = {
            "success": ("var(--green-100)",  "var(--green-800)", "var(--green-800)", "✓"),
            "warning": ("var(--yellow-100)", "var(--yellow-800)", "var(--yellow-800)", "⚠"),
            "error":   ("var(--red-100)",    "var(--red-800)",   "var(--red-800)",   "✕"),
            "info":    ("var(--blue-100)",   "var(--blue-900)",  "var(--blue-500)",  "ℹ"),
        }
        bg, fg, border_c, icon = palettes.get(type, palettes["info"])
        body_html = f'<div class="ds-alert-body" style="color:{fg}">{body}</div>' if body else ""
        st.markdown(
            f"""
            <div class="ds-alert" style="background:{bg};border-color:{border_c}">
                <span class="ds-alert-icon" style="color:{border_c}">{icon}</span>
                <div>
                    <div class="ds-alert-title" style="color:{fg}">{title}</div>
                    {body_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Data table ───────────────────────────────────────────────────────────

    @staticmethod
    def table(
        headers: List[str],
        rows: List[List[Any]],
        caption: Optional[str] = None,
    ):
        """Styled HTML table with hover rows."""
        th_cells = "".join(f"<th>{h}</th>" for h in headers)
        tr_rows = ""
        for row in rows:
            tds = "".join(f"<td>{cell}</td>" for cell in row)
            tr_rows += f"<tr>{tds}</tr>"
        cap_html = (
            f'<caption style="font-size:.75rem;color:var(--text-muted);text-align:left;padding-bottom:.5rem">{caption}</caption>'
            if caption else ""
        )
        st.markdown(
            f"""
            <div class="ds-card" style="padding:0;overflow:hidden">
                <table class="ds-table">
                    {cap_html}
                    <thead><tr>{th_cells}</tr></thead>
                    <tbody>{tr_rows}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Skeleton loader ──────────────────────────────────────────────────────

    @staticmethod
    def skeleton(lines: int = 3, last_short: bool = True):
        """Animated placeholder while content loads."""
        bars = ""
        for i in range(lines):
            short = last_short and i == lines - 1
            w = "55%" if short else "100%"
            h = "14px" if i > 0 else "20px"
            bars += f'<div class="ds-skeleton" style="width:{w};height:{h};margin-bottom:.6rem"></div>'
        st.markdown(
            f'<div style="padding:.5rem 0;animation:fade-up 300ms ease both">{bars}</div>',
            unsafe_allow_html=True,
        )

    # ── Source document card ─────────────────────────────────────────────────

    @staticmethod
    def source_card(filename: str, preview: str = ""):
        preview_html = (
            f'<div style="font-size:.75rem;color:var(--text-muted);margin-top:.25rem">'
            f'{preview[:120]}…</div>'
            if preview else ""
        )
        st.markdown(
            f"""
            <div class="source-document">
                <div style="font-weight:600;font-size:.875rem">📄 {filename}</div>
                {preview_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Toast ────────────────────────────────────────────────────────────────

    @staticmethod
    def toast(message: str, type: str = "info"):
        icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
        st.toast(f"{icons.get(type, 'ℹ️')} {message}")


# ── Chat bubble ───────────────────────────────────────────────────────────────

def chat_bubble(
    content: str,
    role: str = "user",
    timestamp: Optional[datetime] = None,
):
    time_str = (timestamp or datetime.now()).strftime("%I:%M %p")
    align = "flex-end" if role == "user" else "flex-start"
    css_class = "ds-bubble-user" if role == "user" else "ds-bubble-assistant"
    st.markdown(
        f"""
        <div style="display:flex;justify-content:{align};margin:.6rem 0;animation:fade-up 200ms ease both">
            <div class="{css_class}" style="padding:.7rem 1rem;max-width:80%;box-shadow:var(--shadow-sm)">
                <div style="font-size:.875rem;line-height:1.55">{content}</div>
                <div style="font-size:.6rem;opacity:.55;margin-top:.35rem;text-align:right">{time_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── App bootstrap ─────────────────────────────────────────────────────────────

def apply_professional_theme():
    st.set_page_config(
        page_title="Smart RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "<style>#MainMenu,footer,header{visibility:hidden}</style>",
        unsafe_allow_html=True,
    )
    DesignSystem.inject_theme()


# ── Global instance ───────────────────────────────────────────────────────────
ds = DesignSystem()
