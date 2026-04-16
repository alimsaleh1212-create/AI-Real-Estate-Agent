"""Predictions & Insights Monitor — live analytics from Supabase.

Reads the predictions and insights tables written by src/database.py
and renders interactive Plotly charts so you can monitor model usage,
price distributions, feature completeness, and query intent over time.

Shows a friendly empty-state when Supabase is not configured or the
tables have no rows yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import fetch_insights, fetch_predictions
from ui.styles import PRIMARY as ACCENT, SUCCESS as GREEN, ERROR as RED, WARNING, inject_css, apply_plotly_layout

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Predictions Dashboard — AI Real Estate Agent",
    page_icon="🔮",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

inject_css()

col_title, col_refresh, col_back = st.columns([4, 1, 1])
with col_title:
    st.title("🔮 Predictions & Insights Monitor")
    st.caption("Live data from Supabase — all model predictions and market insight queries")
with col_refresh:
    st.write("")
    st.write("")
    if st.button("↺ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_back:
    st.write("")
    st.write("")
    if st.button("← Back to Agent", use_container_width=True):
        st.switch_page("app.py")

st.divider()

# ---------------------------------------------------------------------------
# Data loading (30-second TTL so the page stays live)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner="Loading predictions…")
def _load_predictions() -> pd.DataFrame:
    records = fetch_predictions(limit=500)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


@st.cache_data(ttl=30, show_spinner="Loading insights…")
def _load_insights() -> pd.DataFrame:
    records = fetch_insights(limit=500)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


pred_df = _load_predictions()
ins_df = _load_insights()

# ---------------------------------------------------------------------------
# Empty-state guard
# ---------------------------------------------------------------------------

if pred_df.empty and ins_df.empty:
    st.info(
        "No data yet — make a few predictions or ask market questions in the "
        "AI Real Estate Agent and hit **↺ Refresh** to populate this dashboard."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

total_preds = len(pred_df)
successful_preds = int(pred_df["predicted_price"].notna().sum()) if not pred_df.empty else 0
avg_price = pred_df["predicted_price"].mean() if not pred_df.empty and successful_preds else None
total_insights = len(ins_df)
insight_errors = int(ins_df["error"].notna().sum()) if not ins_df.empty else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Predictions", f"{total_preds:,}")
k2.metric(
    "Successful Predictions",
    f"{successful_preds:,}",
    delta=f"{successful_preds / total_preds * 100:.0f}% success" if total_preds else None,
)
k3.metric("Avg Predicted Price", f"${avg_price:,.0f}" if avg_price else "—")
k4.metric("Insights Queries", f"{total_insights:,}")
k5.metric(
    "Insight Errors",
    f"{insight_errors:,}",
    delta=f"-{insight_errors}" if insight_errors else None,
    delta_color="inverse",
)

st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Predicted Price Distribution
# ---------------------------------------------------------------------------

price_df = pred_df[pred_df["predicted_price"].notna()].copy() if not pred_df.empty else pd.DataFrame()

if not price_df.empty:
    st.subheader("💰 Predicted Price Distribution")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            price_df, x="predicted_price", nbins=30,
            title="Predicted Price Histogram",
            color_discrete_sequence=[ACCENT],
            labels={"predicted_price": "Predicted Price ($)"},
            template="plotly_dark",
        )
        fig.add_vline(
            x=price_df["predicted_price"].median(),
            line_dash="dash", line_color=WARNING,
            annotation_text=f"Median ${price_df['predicted_price'].median():,.0f}",
            annotation_position="top right",
        )
        fig.update_layout(showlegend=False)
        apply_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "neighborhood" in price_df.columns and price_df["neighborhood"].notna().any():
            nbhd = (
                price_df.dropna(subset=["neighborhood"])
                .groupby("neighborhood")["predicted_price"]
                .mean()
                .reset_index()
                .rename(columns={"predicted_price": "Avg Price"})
                .sort_values("Avg Price")
            )
            fig = px.bar(
                nbhd, x="Avg Price", y="neighborhood",
                orientation="h",
                title="Avg Predicted Price by Neighborhood",
                color="Avg Price",
                color_continuous_scale="Plasma",
                text=nbhd["Avg Price"].apply(lambda v: f"${v:,.0f}"),
                labels={"Avg Price": "Avg Predicted Price ($)", "neighborhood": ""},
                template="plotly_dark",
            )
            fig.update_traces(textposition="outside", textfont_color="#8899BB")
            fig.update_layout(coloraxis_showscale=False)
            apply_plotly_layout(fig, margin=dict(t=46, b=10, l=10, r=110))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough neighborhood data to chart yet.")

    st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Predictions Over Time
# ---------------------------------------------------------------------------

if not pred_df.empty and "created_at" in pred_df.columns:
    st.subheader("📈 Prediction Activity Over Time")
    timeline = (
        pred_df.set_index("created_at")
        .resample("D")
        .size()
        .reset_index(name="count")
    )
    if len(timeline) > 1:
        fig = px.area(
            timeline, x="created_at", y="count",
            title="Daily Predictions",
            color_discrete_sequence=[ACCENT],
            labels={"created_at": "Date", "count": "Predictions"},
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False)
        apply_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Queried Property Characteristics
# ---------------------------------------------------------------------------

if not pred_df.empty:
    st.subheader("🏗️ Queried Property Characteristics")
    c1, c2, c3 = st.columns(3)

    with c1:
        col = "overall_qual"
        if col in pred_df.columns and pred_df[col].notna().any():
            fig = px.histogram(
                pred_df[pred_df[col].notna()], x=col,
                title="Overall Quality Distribution",
                color_discrete_sequence=[ACCENT],
                labels={col: "Overall Quality (1–10)"},
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False)
            apply_plotly_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        col = "extracted_count"
        if col in pred_df.columns:
            fig = px.histogram(
                pred_df, x=col,
                title="Features Extracted per Query",
                color_discrete_sequence=[GREEN],
                labels={col: "Features Extracted (out of 10)"},
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False)
            apply_plotly_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    with c3:
        col = "total_sf"
        if col in pred_df.columns and pred_df[col].notna().any():
            fig = px.histogram(
                pred_df[pred_df[col].notna()], x=col,
                title="Total Floor Area Distribution",
                color_discrete_sequence=[WARNING],
                labels={col: "Total Floor Area (sqft)"},
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False)
            apply_plotly_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Insights Activity
# ---------------------------------------------------------------------------

if not ins_df.empty:
    st.subheader("💡 Market Insights Activity")
    c1, c2 = st.columns(2)

    with c1:
        intent_counts = ins_df["intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig = px.pie(
            intent_counts, names="Intent", values="Count",
            title="Query Intent Breakdown",
            color_discrete_sequence=[ACCENT, GREEN],
            template="plotly_dark",
        )
        apply_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ins_df["_status"] = ins_df["error"].isna().map({True: "Success", False: "Error"})
        status_counts = ins_df["_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(
            status_counts, names="Status", values="Count",
            title="Insights Success Rate",
            color="Status",
            color_discrete_map={"Success": GREEN, "Error": RED},
            template="plotly_dark",
        )
        apply_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Recent Activity Tables
# ---------------------------------------------------------------------------

st.subheader("📋 Recent Activity")
tab_pred, tab_ins = st.tabs(["Predictions", "Insights"])

with tab_pred:
    if not pred_df.empty:
        display_cols = [
            "created_at", "query", "predicted_price", "extracted_count",
            "neighborhood", "overall_qual", "total_sf", "year_built", "error",
        ]
        show = [c for c in display_cols if c in pred_df.columns]
        st.dataframe(
            pred_df[show].head(25).rename(columns={
                "created_at": "Timestamp",
                "query": "Query",
                "predicted_price": "Predicted Price ($)",
                "extracted_count": "Features Extracted",
                "neighborhood": "Neighborhood",
                "overall_qual": "Overall Quality",
                "total_sf": "Total sqft",
                "year_built": "Year Built",
                "error": "Error",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions recorded yet.")

with tab_ins:
    if not ins_df.empty:
        display_cols = ["created_at", "query", "intent", "answer", "error"]
        show = [c for c in display_cols if c in ins_df.columns]
        st.dataframe(
            ins_df[show].head(25).rename(columns={
                "created_at": "Timestamp",
                "query": "Query",
                "intent": "Intent",
                "answer": "Answer",
                "error": "Error",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No insights queries recorded yet.")
