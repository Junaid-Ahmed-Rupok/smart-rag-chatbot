"""
Color tokens — theme-aware, dark mode safe.
Use CSS vars in HTML/CSS, Python constants for Plotly/Altair/st.markdown logic.
"""

# ── CSS variable names (use these in HTML strings) ───────────────────────────
# These automatically flip between light and dark via the CSS in components.py.

CSS = {
    "primary":     "var(--blue-500)",
    "primary_dark":"var(--blue-700)",
    "secondary":   "var(--text-muted)",       # neutral; pick a ramp if you need a true accent
    "success":     "var(--green-800)",
    "surface":     "var(--surface)",
    "surface_alt": "var(--surface-alt)",
    "text":        "var(--text-primary)",
    "text_muted":  "var(--text-muted)",
    "border":      "var(--border)",
}

# ── Static hex pairs (light / dark) for Plotly, Altair, PIL, etc. ────────────
# Libraries that can't read CSS variables need real hex values.

LIGHT = {
    "primary":      "#1E88E5",
    "primary_dark": "#1565C0",
    "secondary":    "#64748B",
    "success":      "#065F46",
    "surface":      "#FFFFFF",
    "surface_alt":  "#F8FAFC",
    "text":         "#0F172A",
    "text_muted":   "#64748B",
    "border":       "#E2E8F0",
}

DARK = {
    "primary":      "#60A5FA",
    "primary_dark": "#3B82F6",
    "secondary":    "#94A3B8",
    "success":      "#6EE7B7",
    "surface":      "#1E293B",
    "surface_alt":  "#0F172A",
    "text":         "#F1F5F9",
    "text_muted":   "#94A3B8",
    "border":       "#334155",
}


def get(key: str, dark: bool = False) -> str:
    """Return the hex value for a token in light or dark mode."""
    palette = DARK if dark else LIGHT
    if key not in palette:
        raise KeyError(f"Unknown color token '{key}'. Available: {list(palette)}")
    return palette[key]


def is_dark_mode() -> bool:
    """Best-effort detection of Streamlit dark theme."""
    try:
        import streamlit as st
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def resolve(key: str) -> str:
    """Return the correct hex for the current Streamlit theme."""
    return get(key, dark=is_dark_mode())
