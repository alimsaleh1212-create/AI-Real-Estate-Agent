"""Dataset Analytics Dashboard — Ames Housing, 10 Selected Features.

Visualises the statistical relationships that drove feature selection,
including price distributions, correlations, quality breakdowns,
neighbourhood medians, and trained-model feature importance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_PROCESSED_DIR, MODEL_PATH, SELECTED_FEATURES
from ui.styles import ACCENT, inject_css, apply_plotly_layout

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Trained Data Dashboards — AI Real Estate Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ORDINAL_ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]
_ORDINAL_LABELS = {
    "Po": "Poor", "Fa": "Fair", "TA": "Typical",
    "Gd": "Good", "Ex": "Excellent",
}
_BSMT_ORDER = ["None"] + _ORDINAL_ORDER
_BSMT_LABELS = {"None": "No Basement", **_ORDINAL_LABELS}

_PALETTE = px.colors.sequential.Plasma_r


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def _load_data() -> pd.DataFrame:
    """Combine all splits into one frame with selected features + SalePrice."""
    frames = []
    for split in ("train", "val", "test"):
        X = pd.read_csv(DATA_PROCESSED_DIR / f"X_{split}.csv", index_col=0)
        y = pd.read_csv(DATA_PROCESSED_DIR / f"y_{split}.csv", index_col=0)
        X = X[SELECTED_FEATURES].copy()
        X["SalePrice"] = y.squeeze().values
        frames.append(X)
    return pd.concat(frames, ignore_index=True)


@st.cache_resource(show_spinner="Loading model…")
def _load_pipeline() -> object:
    """Load the trained sklearn Pipeline."""
    return joblib.load(MODEL_PATH)


def _feature_importances(pipeline: object) -> pd.DataFrame:
    """Extract per-original-feature importances (aggregates OHE columns)."""
    pre = pipeline.named_steps["preprocessor"]  # type: ignore[attr-defined]
    model = pipeline.named_steps["model"]  # type: ignore[attr-defined]

    raw_names: list[str] = list(pre.get_feature_names_out())
    raw_imp: np.ndarray = model.feature_importances_

    rows: dict[str, float] = {}
    for name, imp in zip(raw_names, raw_imp):
        # ColumnTransformer prefixes: "num__X", "ord__X", "nom__X_value"
        original = name.split("__", 1)[1].split("_")[0]
        rows[original] = rows.get(original, 0.0) + float(imp)

    df = (
        pd.DataFrame(rows.items(), columns=["Feature", "Importance"])
        .sort_values("Importance", ascending=True)
    )
    return df


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

inject_css()

col_title, col_back = st.columns([5, 1])
with col_title:
    st.title("📊 Trained Data Dashboards")
    st.caption(
        "Ames, Iowa — 2,929 homes · 10 selected predictors · "
        "GradientBoosting winner (CV RMSE 0.1468)"
    )
with col_back:
    st.write("")
    st.write("")
    if st.button("← Back to Agent", use_container_width=True):
        st.switch_page("app.py")

st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = _load_data()
pipeline = _load_pipeline()

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Homes", f"{len(df):,}")
k2.metric("Median Price", f"${df['SalePrice'].median():,.0f}")
k3.metric("Mean Price", f"${df['SalePrice'].mean():,.0f}")
price_range = f"${df['SalePrice'].min():,.0f} – ${df['SalePrice'].max():,.0f}"
k4.metric("Price Range", price_range)
k5.metric("Selected Features", "10")

st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Price Distribution
# ---------------------------------------------------------------------------

st.subheader("💰 Sale Price Distribution")
c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(
        df, x="SalePrice", nbins=60,
        title="Sale Price Histogram",
        color_discrete_sequence=[ACCENT],
        labels={"SalePrice": "Sale Price ($)"},
        template="plotly_dark",
    )
    fig.add_vline(
        x=df["SalePrice"].median(), line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Median ${df['SalePrice'].median():,.0f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=df["SalePrice"].mean(), line_dash="dot", line_color="#34D399",
        annotation_text=f"Mean ${df['SalePrice'].mean():,.0f}",
        annotation_position="top left",
    )
    fig.update_layout(showlegend=False)
    apply_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.box(
        df, y="SalePrice", x="OverallQual",
        title="Sale Price by Overall Quality",
        color="OverallQual",
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        labels={"SalePrice": "Sale Price ($)", "OverallQual": "Overall Quality"},
        template="plotly_dark",
    )
    fig.update_layout(showlegend=False)
    apply_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2 — Correlation Heatmap
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔗 Feature Correlation Matrix")

numeric_cols = ["OverallQual", "TotalSF", "GarageCars", "TotalBath",
                "YearBuilt", "TotalBsmtSF", "SalePrice"]
corr = df[numeric_cols].corr().round(2)

fig = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu",
    zmid=0,
    text=corr.values,
    texttemplate="%{text}",
    textfont={"size": 12, "color": "#8899BB"},
    hovertemplate="%{x} vs %{y}: %{z}<extra></extra>",
))
fig.update_layout(
    title="Pearson Correlation — Numeric Selected Features + Sale Price",
    height=420,
)
apply_plotly_layout(fig)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Numeric Features vs Sale Price
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📈 Numeric Features vs Sale Price")

numeric_features = ["OverallQual", "TotalSF", "TotalBsmtSF",
                    "GarageCars", "TotalBath", "YearBuilt"]
feat_labels = {
    "OverallQual": "Overall Quality (1–10)",
    "TotalSF": "Total Floor Area (sqft)",
    "TotalBsmtSF": "Basement Area (sqft)",
    "GarageCars": "Garage Capacity (cars)",
    "TotalBath": "Total Bathrooms",
    "YearBuilt": "Year Built",
}

cols = st.columns(3)
for i, feat in enumerate(numeric_features):
    with cols[i % 3]:
        fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=42),
            x=feat, y="SalePrice",
            trendline="ols",
            title=feat_labels[feat],
            labels={feat: feat_labels[feat], "SalePrice": "Sale Price ($)"},
            color_discrete_sequence=[ACCENT],
            trendline_color_override="#34D399",
            template="plotly_dark",
            opacity=0.45,
        )
        fig.update_layout(showlegend=False)
        apply_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4 — Quality Features vs Sale Price (Box Plots)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🏆 Quality Features vs Sale Price")

q1, q2, q3 = st.columns(3)

for col_widget, feat, order, labels, title in [
    (q1, "KitchenQual", _ORDINAL_ORDER, _ORDINAL_LABELS, "Kitchen Quality"),
    (q2, "ExterQual", _ORDINAL_ORDER, _ORDINAL_LABELS, "Exterior Quality"),
    (q3, "BsmtQual", _BSMT_ORDER, _BSMT_LABELS, "Basement Quality"),
]:
    with col_widget:
        plot_df = df[df[feat].isin(order)].copy()
        plot_df[feat] = plot_df[feat].map(labels)
        ordered_labels = [labels[o] for o in order if o in labels]
        fig = px.box(
            plot_df, x=feat, y="SalePrice",
            category_orders={feat: ordered_labels},
            title=title,
            color=feat,
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            labels={feat: title, "SalePrice": "Sale Price ($)"},
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False)
        apply_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5 — Neighbourhood Median Prices
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🏘️ Median Sale Price by Neighbourhood")

nbhd = (
    df.groupby("Neighborhood")["SalePrice"]
    .agg(["median", "count"])
    .reset_index()
    .rename(columns={"median": "Median Price", "count": "Count"})
    .sort_values("Median Price")
)

fig = px.bar(
    nbhd, x="Median Price", y="Neighborhood",
    orientation="h",
    color="Median Price",
    color_continuous_scale="Plasma",
    text=nbhd["Median Price"].apply(lambda v: f"${v:,.0f}"),
    title="Neighbourhood Median Sale Price (sorted ascending)",
    labels={"Median Price": "Median Sale Price ($)", "Neighborhood": ""},
    template="plotly_dark",
    height=640,
)
fig.update_traces(textposition="outside", textfont_color="#8899BB")
fig.update_layout(coloraxis_showscale=False)
apply_plotly_layout(fig, margin=dict(t=46, b=10, l=10, r=130))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 6 — Model Feature Importance
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🤖 Model Feature Importance (GradientBoosting)")

try:
    imp_df = _feature_importances(pipeline)
    imp_df["Importance %"] = (imp_df["Importance"] * 100).round(1)

    fig = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Plasma",
        text=imp_df["Importance %"].apply(lambda v: f"{v:.1f}%"),
        title="Feature Importance — contribution to model predictions",
        labels={"Importance": "Importance Score", "Feature": ""},
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside", textfont_color="#8899BB")
    fig.update_layout(coloraxis_showscale=False)
    apply_plotly_layout(fig, margin=dict(t=46, b=10, l=10, r=90))
    st.plotly_chart(fig, use_container_width=True)

    top = imp_df.iloc[-1]
    st.info(
        f"**{top['Feature']}** is the strongest predictor "
        f"({top['Importance %']:.1f}% of model decisions), "
        "consistent with domain knowledge that overall quality drives price more "
        "than any single room or structural feature."
    )
except Exception as exc:
    st.warning(f"Could not load feature importance: {exc}")
