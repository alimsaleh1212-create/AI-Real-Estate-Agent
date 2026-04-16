"""Shared design system for all Streamlit pages — "Clarity Dark" theme.

One primary color (indigo #6366F1), zinc-950 base, Inter font throughout.
Every design decision prioritises readability and professional clarity.

Usage:
    from ui.styles import inject_css, apply_plotly_layout, PRIMARY, SUCCESS, ERROR

    inject_css()          # after set_page_config, before any other st.* call
    fig = px.bar(...)
    apply_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ---------------------------------------------------------------------------
# Palette — one accent, zinc neutrals, semantic colours
# ---------------------------------------------------------------------------

PRIMARY   = "#6366F1"   # indigo-500  — buttons, active states, highlights
PRIMARY_L = "#818CF8"   # indigo-400  — hover states, lighter accents
SUCCESS   = "#22C55E"   # green-500
ERROR     = "#EF4444"   # red-500
WARNING   = "#F59E0B"   # amber-500
MUTED     = "#A1A1AA"   # zinc-400    — secondary text, labels
BORDER    = "#3F3F46"   # zinc-700    — visible borders
SURFACE   = "#18181B"   # zinc-900    — card / form background
SURFACE_2 = "#27272A"   # zinc-800    — elevated elements

# Legacy aliases kept so existing page code doesn't break
ACCENT = PRIMARY
GREEN  = SUCCESS
RED    = ERROR

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── GLOBAL ──────────────────────────────────── */
.stApp, html, body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #09090B !important;
}

.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1280px !important;
}

footer, #MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── TYPOGRAPHY ──────────────────────────────── */
h1 {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 1.875rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: #FAFAFA !important;
    line-height: 1.2 !important;
    margin-bottom: 0.25rem !important;
}

h2 {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    color: #E4E4E7 !important;
    margin-top: 0.5rem !important;
}

h3 {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #71717A !important;
    margin-bottom: 0.75rem !important;
}

p, .stMarkdown p, .stMarkdown li {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #A1A1AA !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
}

/* ── DIVIDERS ────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #27272A !important;
    margin: 1.5rem 0 !important;
}

/* ── METRIC CARDS ────────────────────────────── */
[data-testid="stMetric"] {
    background: #18181B !important;
    border: 1px solid #27272A !important;
    border-radius: 10px !important;
    padding: 1.25rem 1.5rem !important;
    transition: border-color 0.15s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: #3F3F46 !important;
}

[data-testid="stMetricLabel"] > div {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #71717A !important;
}

[data-testid="stMetricValue"] > div {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums !important;
    color: #FAFAFA !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stMetricDelta"] > div {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ── BUTTONS ─────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    background: #18181B !important;
    color: #E4E4E7 !important;
    border: 1px solid #3F3F46 !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0 !important;
}

.stButton > button:hover {
    background: #27272A !important;
    border-color: #6366F1 !important;
    color: #FAFAFA !important;
    transform: translateY(-1px) !important;
}

/* Primary / submit */
button[kind="primaryFormSubmit"],
.stButton > button[kind="primary"] {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(99,102,241,0.3) !important;
}

button[kind="primaryFormSubmit"]:hover,
.stButton > button[kind="primary"]:hover {
    background: #4F46E5 !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── TEXT INPUT / TEXT AREA ──────────────────── */
.stTextArea textarea {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.9rem !important;
    background: #18181B !important;
    border: 1px solid #3F3F46 !important;
    border-radius: 8px !important;
    color: #F4F4F5 !important;
    line-height: 1.6 !important;
    transition: border-color 0.15s ease !important;
}

.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

.stTextArea label p {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #71717A !important;
}

/* ── FORM CONTAINER ──────────────────────────── */
[data-testid="stForm"] {
    background: #18181B !important;
    border: 1px solid #27272A !important;
    border-radius: 12px !important;
    padding: 1.75rem 2rem !important;
}

/* ── SELECTBOX ───────────────────────────────── */
.stSelectbox > div > div {
    background: #18181B !important;
    border: 1px solid #3F3F46 !important;
    border-radius: 8px !important;
    color: #F4F4F5 !important;
}

.stSelectbox label p {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #71717A !important;
}

/* ── NUMBER INPUT ────────────────────────────── */
.stNumberInput input {
    background: #18181B !important;
    border: 1px solid #3F3F46 !important;
    border-radius: 8px !important;
    color: #F4F4F5 !important;
    font-variant-numeric: tabular-nums !important;
}

.stNumberInput label p {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #71717A !important;
}

/* ── SLIDER ──────────────────────────────────── */
.stSlider label p {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #71717A !important;
}

/* ── TABS ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #18181B !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: 1px solid #27272A !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #71717A !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s ease !important;
}

.stTabs [aria-selected="true"] {
    background: #27272A !important;
    color: #FAFAFA !important;
}

/* ── EXPANDER ────────────────────────────────── */
[data-testid="stExpander"] {
    background: #18181B !important;
    border: 1px solid #27272A !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #A1A1AA !important;
}

/* ── ALERTS ──────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-width: 1px !important;
}

/* ── CAPTION ─────────────────────────────────── */
.stCaption p {
    color: #71717A !important;
    font-size: 0.8rem !important;
}

/* ── DATAFRAME ───────────────────────────────── */
.stDataFrame {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #27272A !important;
}

/* ── SIDEBAR ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #18181B !important;
    border-right: 1px solid #27272A !important;
}

[data-testid="stSidebarNav"] a {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #71717A !important;
    border-radius: 6px !important;
    transition: all 0.12s ease !important;
}

[data-testid="stSidebarNav"] a:hover {
    color: #E4E4E7 !important;
    background: #27272A !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(99,102,241,0.12) !important;
    color: #818CF8 !important;
}

/* ── PLOTLY CHART FRAME ──────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #27272A !important;
}

/* ── SCROLLBAR ───────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #09090B; }
::-webkit-scrollbar-thumb { background: #3F3F46; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366F1; }
</style>
"""

# ---------------------------------------------------------------------------
# Plotly layout — transparent background, zinc grid, Inter font
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#71717A", size=12),
    title_font=dict(
        family="Inter, system-ui, sans-serif", color="#A1A1AA", size=13,
    ),
    xaxis=dict(
        gridcolor="#27272A",
        linecolor="#27272A",
        tickfont=dict(color="#71717A", size=11),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#27272A",
        linecolor="#27272A",
        tickfont=dict(color="#71717A", size=11),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(24,24,27,0.9)",
        bordercolor="#3F3F46",
        borderwidth=1,
        font=dict(color="#A1A1AA", size=11),
    ),
    margin=dict(t=40, b=12, l=12, r=12),
    hoverlabel=dict(
        bgcolor="#27272A",
        bordercolor="#3F3F46",
        font=dict(family="Inter, system-ui, sans-serif", color="#F4F4F5", size=12),
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color="#71717A"),
        title_font=dict(color="#71717A"),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#27272A",
        borderwidth=1,
    ),
)

# Chart colour sequences — indigo-led, harmonious
CHART_SEQ   = ["#6366F1", "#818CF8", "#A5B4FC", "#C7D2FE"]
CHART_MULTI = ["#6366F1", "#22C55E", "#F59E0B", "#EF4444", "#06B6D4", "#EC4899"]


def inject_css() -> None:
    """Inject the shared Clarity Dark CSS into the current page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def apply_plotly_layout(fig: object, **overrides: object) -> None:
    """Apply the shared Plotly theme to a figure in place.

    Args:
        fig: Any Plotly Figure.
        **overrides: Extra layout kwargs merged on top of the defaults.
    """
    import plotly.graph_objects as go  # local — ui-only dependency

    assert isinstance(fig, go.Figure)
    fig.update_layout(**{**PLOTLY_LAYOUT, **overrides})
    fig.update_xaxes(gridcolor="#27272A", linecolor="#27272A",
                     tickfont=dict(color="#71717A", size=11))
    fig.update_yaxes(gridcolor="#27272A", linecolor="#27272A",
                     tickfont=dict(color="#71717A", size=11))
