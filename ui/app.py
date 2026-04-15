"""Streamlit frontend for the AI Real Estate Agent.

Two-step prediction flow:
    Step 1 — User enters a natural-language property description.
             LLM Stage 1 extracts features; user reviews and fills gaps.
    Step 2 — User submits the complete feature set.
             ML model predicts price; LLM Stage 2 generates interpretation.

Bonus: if the query is an analysis question (intent = 'analysis'),
the UI shows a market-insights answer instead of the prediction flow.

Calls src functions directly (no HTTP) so it can run standalone without
the FastAPI server being up.
"""

import logging

import streamlit as st

from src.config import FEATURE_DEFINITIONS, ORDINAL_ORDERS
from src.llm_chain import (
    ExtractionError,
    classify_intent,
    extract_features,
    generate_market_insights,
    predict_and_interpret,
)
from src.predictor import get_stats, load_model, load_stats, predict_price
from src.schemas import ExtractedFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Real Estate Agent — Ames, Iowa",
    page_icon="🏠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# One-time resource loading (cached per process — model loaded once)
# ---------------------------------------------------------------------------

_FEATURE_LABELS: dict[str, str] = {
    "OverallQual": "Overall Quality (1–10)",
    "TotalSF": "Total Floor Area (sqft)",
    "GarageCars": "Garage Capacity (cars)",
    "TotalBath": "Total Bathrooms",
    "YearBuilt": "Year Built",
    "TotalBsmtSF": "Basement Area (sqft)",
    "KitchenQual": "Kitchen Quality",
    "BsmtQual": "Basement Quality",
    "ExterQual": "Exterior Material Quality",
    "Neighborhood": "Neighborhood",
}

_PROMPT_EXAMPLE = (
    "**Example:** \"3-bedroom, 2-story home in Northridge Heights, built in 1998. "
    "Total living area is 2,100 sqft with a 900 sqft finished basement "
    "in good condition. "
    "2-car garage, 2.5 bathrooms, excellent kitchen, good exterior finish, "
    "overall quality rating 8 out of 10.\""
)

_ORDINAL_VALUES = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
_QUALITY_LABELS = {
    "None": "None (no basement)",
    "Po": "Po — Poor",
    "Fa": "Fa — Fair",
    "TA": "TA — Typical/Average",
    "Gd": "Gd — Good",
    "Ex": "Ex — Excellent",
}
_ORDINAL_DISPLAY = [_QUALITY_LABELS[v] for v in _ORDINAL_VALUES]
_NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]


@st.cache_resource(show_spinner="Loading model…")
def _init_predictor() -> None:
    """Load model and stats once per Streamlit server process."""
    load_model()
    load_stats()


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _reset_state() -> None:
    """Clear prediction state so the user can start over."""
    for key in ("step", "extracted", "query"):
        st.session_state.pop(key, None)


def _init_state() -> None:
    """Initialise session state on first run."""
    if "step" not in st.session_state:
        st.session_state["step"] = "input"


# ---------------------------------------------------------------------------
# Gap-filling form helpers
# ---------------------------------------------------------------------------

def _ordinal_index(current: str | None, options: list[str]) -> int:
    """Return the selectbox index for a current value (or 0 if None)."""
    if current is None:
        return 0
    try:
        return options.index(current)
    except ValueError:
        return 0


def _render_gap_form(extracted: ExtractedFeatures) -> ExtractedFeatures:
    """Render a form for all features; pre-fill with extracted values.

    Args:
        extracted: Stage 1 output with some None fields.

    Returns:
        Updated ExtractedFeatures with user-supplied gap values merged in.
    """
    updates: dict[str, object] = {}

    st.subheader("Feature Review & Gap Filling")
    st.caption(
        f"Extracted **{len(extracted.extracted_features)}/10** features. "
        "Review extracted values and fill in any missing ones."
    )

    with st.form("gap_form"):
        col1, col2 = st.columns(2)

        with col1:
            updates["OverallQual"] = st.slider(
                _FEATURE_LABELS["OverallQual"],
                min_value=1, max_value=10,
                value=extracted.OverallQual or 6,
                help=FEATURE_DEFINITIONS["OverallQual"]["description"],
            )
            updates["TotalSF"] = st.number_input(
                _FEATURE_LABELS["TotalSF"],
                min_value=300, max_value=12000,
                value=int(extracted.TotalSF or 1500),
                step=50,
                help=FEATURE_DEFINITIONS["TotalSF"]["description"],
            )
            updates["GarageCars"] = st.selectbox(
                _FEATURE_LABELS["GarageCars"],
                options=[0, 1, 2, 3, 4],
                index=extracted.GarageCars if extracted.GarageCars is not None else 2,
                help=FEATURE_DEFINITIONS["GarageCars"]["description"],
            )
            updates["TotalBath"] = st.number_input(
                _FEATURE_LABELS["TotalBath"],
                min_value=0.0, max_value=8.0,
                value=float(extracted.TotalBath or 2.0),
                step=0.5,
                help=FEATURE_DEFINITIONS["TotalBath"]["description"],
            )
            updates["YearBuilt"] = st.slider(
                _FEATURE_LABELS["YearBuilt"],
                min_value=1872, max_value=2010,
                value=extracted.YearBuilt or 1990,
                help=FEATURE_DEFINITIONS["YearBuilt"]["description"],
            )

        with col2:
            updates["TotalBsmtSF"] = st.number_input(
                _FEATURE_LABELS["TotalBsmtSF"],
                min_value=0, max_value=7000,
                value=int(extracted.TotalBsmtSF or 0),
                step=50,
                help=FEATURE_DEFINITIONS["TotalBsmtSF"]["description"],
            )
            kq_idx = _ordinal_index(extracted.KitchenQual, _ORDINAL_VALUES[1:])
            updates["KitchenQual"] = _ORDINAL_VALUES[1:][
                st.selectbox(
                    _FEATURE_LABELS["KitchenQual"],
                    options=range(len(_ORDINAL_VALUES[1:])),
                    format_func=lambda i: _QUALITY_LABELS[_ORDINAL_VALUES[1:][i]],
                    index=kq_idx,
                    help=FEATURE_DEFINITIONS["KitchenQual"]["description"],
                )
            ]
            bq_options = ORDINAL_ORDERS["BsmtQual"]
            bq_idx = _ordinal_index(extracted.BsmtQual, bq_options)
            updates["BsmtQual"] = bq_options[
                st.selectbox(
                    _FEATURE_LABELS["BsmtQual"],
                    options=range(len(bq_options)),
                    format_func=lambda i: _QUALITY_LABELS[bq_options[i]],
                    index=bq_idx,
                    help=FEATURE_DEFINITIONS["BsmtQual"]["description"],
                )
            ]
            eq_idx = _ordinal_index(extracted.ExterQual, _ORDINAL_VALUES[1:])
            updates["ExterQual"] = _ORDINAL_VALUES[1:][
                st.selectbox(
                    _FEATURE_LABELS["ExterQual"],
                    options=range(len(_ORDINAL_VALUES[1:])),
                    format_func=lambda i: _QUALITY_LABELS[_ORDINAL_VALUES[1:][i]],
                    index=eq_idx,
                    help=FEATURE_DEFINITIONS["ExterQual"]["description"],
                )
            ]
            nbhd_idx = (
                _NEIGHBORHOODS.index(extracted.Neighborhood)
                if extracted.Neighborhood in _NEIGHBORHOODS
                else 0
            )
            updates["Neighborhood"] = st.selectbox(
                _FEATURE_LABELS["Neighborhood"],
                options=_NEIGHBORHOODS,
                index=nbhd_idx,
                help=FEATURE_DEFINITIONS["Neighborhood"]["description"],
            )

        submitted = st.form_submit_button(
            "Get Price Prediction", type="primary", use_container_width=True
        )

    if submitted:
        merged = extracted.model_copy(update=updates)
        return merged
    return extracted


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the full Streamlit application."""
    _init_predictor()
    _init_state()

    col_hdr, col_nav = st.columns([4, 1])
    with col_hdr:
        st.title("AI Real Estate Agent")
        st.caption(
            "Ames, Iowa — describe a property in plain English "
            "for an AI-powered price estimate."
        )
    with col_nav:
        st.write("")
        st.write("")
        if st.button("📊 Dataset Analytics", use_container_width=True):
            st.switch_page("pages/dashboard.py")

    step = st.session_state["step"]

    # ------------------------------------------------------------------
    # Step 1: Query input
    # ------------------------------------------------------------------
    if step == "input":
        query = st.text_area(
            "Property Description",
            placeholder=(
                "Describe the property in plain English — include size, quality, "
                "year built, garage, bathrooms, basement, and neighborhood."
            ),
            height=100,
        )
        st.info(_PROMPT_EXAMPLE)
        col_analyze, col_tip = st.columns([2, 3])
        with col_analyze:
            analyze = st.button("Analyze", type="primary", use_container_width=True)
        with col_tip:
            st.caption(
                "You can also ask market questions like "
                "'What is the average price in Ames?'"
            )

        if analyze and query.strip():
            with st.spinner("Classifying intent…"):
                intent = classify_intent(query)

            if intent == "analysis":
                with st.spinner("Generating market insights…"):
                    try:
                        stats = get_stats()
                        answer = generate_market_insights(query, stats)
                        st.info(answer)
                    except Exception as exc:
                        st.error(f"Could not generate insights: {exc}")
            else:
                with st.spinner("Extracting property features with AI…"):
                    try:
                        extracted = extract_features(query)
                        st.session_state["extracted"] = extracted
                        st.session_state["query"] = query
                        st.session_state["step"] = "fill_gaps"
                        st.rerun()
                    except ExtractionError as exc:
                        st.error(
                            f"Could not extract features: {exc}. "
                            "Try being more specific."
                        )
                    except Exception as exc:
                        st.error(f"An unexpected error occurred: {exc}")

        elif analyze:
            st.warning("Please enter a property description first.")

    # ------------------------------------------------------------------
    # Step 2: Gap filling + prediction
    # ------------------------------------------------------------------
    elif step == "fill_gaps":
        # Retrieve from session state; types set in step "input" above
        extracted = st.session_state["extracted"]
        query = st.session_state["query"]

        st.markdown(f"**Query:** {query}")

        with st.expander("Extracted features", expanded=True):
            ext_names = set(extracted.extracted_features)
            rows = []
            for feat in extracted.model_fields:
                if feat == "confidence":
                    continue
                val = getattr(extracted, feat)
                status = "✅ extracted" if feat in ext_names else "⬜ missing"
                rows.append({
                        "Feature": _FEATURE_LABELS.get(feat, feat),
                        "Value": val,
                        "Status": status,
                    })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        updated = _render_gap_form(extracted)

        if updated is not extracted:
            # Form was submitted — run prediction
            with st.spinner("Predicting price…"):
                try:
                    price = predict_price(updated)
                    stats = get_stats()
                    interpretation = predict_and_interpret(updated, price, stats)

                    sp = stats.get("sale_price_stats", stats)
                    median = sp["median"]
                    pct_vs_median = (price - median) / median * 100

                    st.divider()
                    st.subheader("Price Prediction")
                    col_price, col_diff = st.columns(2)
                    with col_price:
                        st.metric("Estimated Price", f"${price:,.0f}")
                    with col_diff:
                        st.metric(
                            "vs. Ames Median",
                            f"${median:,.0f}",
                            delta=f"{pct_vs_median:+.1f}%",
                        )

                    st.markdown("### Interpretation")
                    st.write(interpretation)

                    if st.button("Start Over"):
                        _reset_state()
                        st.rerun()

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        if st.button("← Back"):
            _reset_state()
            st.rerun()


if __name__ == "__main__":
    main()
