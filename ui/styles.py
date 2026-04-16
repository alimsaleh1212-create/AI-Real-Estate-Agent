"""Shared design system for all Streamlit pages.

Provides a single source of truth for colours, typography, CSS injection,
and Plotly layout overrides so every page shares the same "Data Noir" aesthetic:
  • Deep navy backgrounds  (#080C17 / #0C1424)
  • Warm amber-gold accent (#E8B84B)
  • Cormorant Garamond display headlines
  • DM Sans body copy

Usage:
    from ui.styles import inject_css, apply_plotly_layout, ACCENT, GREEN

    inject_css()          # call once near the top of each page, after set_page_config
    fig = px.bar(...)
    apply_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ---------------------------------------------------------------------------
# Palette constants (import these instead of hardcoding hex strings)
# ---------------------------------------------------------------------------

ACCENT = "#E8B84B"       # warm amber-gold
GREEN = "#34D399"        # emerald success
RED = "#F87171"          # soft red error
INDIGO = "#818CF8"       # indigo secondary
TEAL = "#2DD4BF"         # teal tertiary
MUTED = "#4D6080"        # muted label text
SURFACE = "#0C1424"      # card / form background
BORDER = "#1A2843"       # subtle borders

# ---------------------------------------------------------------------------
# CSS — injected once per page via inject_css()
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,700&display=swap');

/* ── APP SHELL ─────────────────────────────────────────────────── */
.stApp {
    background: #080C17;
    background-image:
        radial-gradient(ellipse at 15% 60%, rgba(232,184,75,0.045) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 10%, rgba(129,140,248,0.04) 0%, transparent 50%);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding: 2.5rem 3rem 5rem !important;
    max-width: 1280px !important;
}

/* ── TYPOGRAPHY ────────────────────────────────────────────────── */
h1 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    color: #F0F4FF !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
    margin-bottom: 0.15rem !important;
}

h2 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 1.65rem !important;
    color: #D8E4F8 !important;
    letter-spacing: -0.02em !important;
    margin-top: 0.25rem !important;
}

h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    color: #4D6080 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.5rem !important;
}

p, .stMarkdown p, .stMarkdown li {
    color: #8899BB !important;
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.7 !important;
}

/* ── DIVIDERS ──────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #14203A !important;
    margin: 2rem 0 !important;
}

/* ── METRIC CARDS ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(140deg, #0D1728 0%, #111E38 100%) !important;
    border: 1px solid #172338 !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.6rem !important;
    position: relative !important;
    overflow: hidden !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(232,184,75,0.3) !important;
    box-shadow: 0 8px 32px rgba(232,184,75,0.08) !important;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #E8B84B55 50%, transparent 100%);
}

[data-testid="stMetricLabel"] > div {
    color: #3D5070 !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stMetricValue"] > div {
    color: #EEF2FF !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums !important;
    letter-spacing: -0.025em !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stMetricDelta"] > div {
    font-size: 0.78rem !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── BUTTONS ───────────────────────────────────────────────────── */
.stButton > button {
    background: #0C1424 !important;
    color: #E8B84B !important;
    border: 1px solid rgba(232,184,75,0.22) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
}

.stButton > button:hover {
    background: rgba(232,184,75,0.07) !important;
    border-color: rgba(232,184,75,0.55) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(232,184,75,0.14) !important;
}

/* Primary / form-submit */
button[kind="primaryFormSubmit"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E8B84B 0%, #C89A35 100%) !important;
    color: #080C17 !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    box-shadow: 0 4px 18px rgba(232,184,75,0.28) !important;
}

button[kind="primaryFormSubmit"]:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #F0C55A 0%, #D4A83C 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(232,184,75,0.38) !important;
}

/* ── TEXT AREA ─────────────────────────────────────────────────── */
.stTextArea textarea {
    background: #0A1020 !important;
    border: 1px solid #172338 !important;
    border-radius: 10px !important;
    color: #D8E4F8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    caret-color: #E8B84B;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextArea textarea:focus {
    border-color: rgba(232,184,75,0.45) !important;
    box-shadow: 0 0 0 3px rgba(232,184,75,0.07) !important;
}

.stTextArea label p {
    color: #3D5070 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* ── FORM CONTAINER ────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: linear-gradient(160deg, #0A1020 0%, #0D1628 100%) !important;
    border: 1px solid #172338 !important;
    border-radius: 16px !important;
    padding: 2rem 2.5rem !important;
}

/* ── SELECTBOX ─────────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: #0A1020 !important;
    border: 1px solid #172338 !important;
    border-radius: 8px !important;
    color: #D8E4F8 !important;
}

.stSelectbox label p {
    color: #3D5070 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* ── NUMBER INPUT ──────────────────────────────────────────────── */
.stNumberInput input {
    background: #0A1020 !important;
    border: 1px solid #172338 !important;
    border-radius: 8px !important;
    color: #D8E4F8 !important;
    font-variant-numeric: tabular-nums !important;
}

.stNumberInput label p {
    color: #3D5070 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* ── SLIDER ────────────────────────────────────────────────────── */
.stSlider label p {
    color: #3D5070 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #E8B84B !important;
    border-color: #E8B84B !important;
}

/* ── TABS ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0A1020 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 3px !important;
    border: 1px solid #172338 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #3D5070 !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.03em !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #152040, #1A2852) !important;
    color: #E8B84B !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35) !important;
}

/* ── ALERTS ────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

[data-testid="stAlert"] p {
    color: inherit !important;
}

/* ── EXPANDER ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0A1020 !important;
    border: 1px solid #172338 !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #8899BB !important;
    font-size: 0.88rem !important;
}

/* ── DATAFRAME ─────────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #172338 !important;
}

/* ── SIDEBAR NAV ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #080C17 !important;
    border-right: 1px solid #172338 !important;
}

[data-testid="stSidebarNav"] a {
    font-family: 'DM Sans', sans-serif !important;
    color: #4D6080 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}

[data-testid="stSidebarNav"] a:hover {
    color: #8899BB !important;
    background: rgba(255,255,255,0.04) !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(232,184,75,0.09) !important;
    color: #E8B84B !important;
}

/* ── CAPTION ───────────────────────────────────────────────────── */
.stCaption p {
    color: #3D5070 !important;
    font-size: 0.78rem !important;
}

/* ── SCROLLBAR ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080C17; }
::-webkit-scrollbar-thumb { background: #172338; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(232,184,75,0.35); }

/* ── PLOTLY CHART CONTAINERS ───────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #172338 !important;
    background: #0A1020 !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Plotly layout overrides — apply to every figure via apply_plotly_layout()
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="rgba(10, 16, 32, 0)",
    plot_bgcolor="rgba(10, 16, 32, 0)",
    font=dict(family="DM Sans, sans-serif", color="#4D6080", size=12),
    title_font=dict(family="DM Sans, sans-serif", color="#8899BB", size=13),
    title_pad=dict(t=4, b=8),
    xaxis=dict(
        gridcolor="#172338",
        gridwidth=1,
        linecolor="#172338",
        tickcolor="#3D5070",
        tickfont=dict(color="#3D5070", size=11, family="DM Sans, sans-serif"),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#172338",
        gridwidth=1,
        linecolor="#172338",
        tickcolor="#3D5070",
        tickfont=dict(color="#3D5070", size=11, family="DM Sans, sans-serif"),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(10,16,32,0.85)",
        bordercolor="#172338",
        borderwidth=1,
        font=dict(color="#8899BB", size=11, family="DM Sans, sans-serif"),
    ),
    margin=dict(t=42, b=12, l=12, r=12),
    hoverlabel=dict(
        bgcolor="#0D1728",
        bordercolor="rgba(232,184,75,0.35)",
        font=dict(family="DM Sans, sans-serif", color="#D8E4F8", size=12),
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color="#4D6080"),
        title_font=dict(color="#4D6080"),
        bgcolor="rgba(10,16,32,0)",
        bordercolor="#172338",
        borderwidth=1,
    ),
)


def inject_css() -> None:
    """Inject the shared Data Noir CSS into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def apply_plotly_layout(fig: object, **overrides: object) -> None:
    """Apply the shared Plotly dark theme to a figure in place.

    Args:
        fig: Any Plotly Figure object.
        **overrides: Additional layout kwargs that take precedence.
    """
    import plotly.graph_objects as go  # local import — ui dependency only

    assert isinstance(fig, go.Figure)
    layout = {**PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**layout)
    # Ensure axis styling propagates to any secondary axes too
    fig.update_xaxes(
        gridcolor="#172338", linecolor="#172338",
        tickfont=dict(color="#3D5070", size=11),
    )
    fig.update_yaxes(
        gridcolor="#172338", linecolor="#172338",
        tickfont=dict(color="#3D5070", size=11),
    )
