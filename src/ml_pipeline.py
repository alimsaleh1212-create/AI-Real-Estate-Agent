"""ML pipeline module for the AI Real Estate Agent.

Extracts training logic from Notebook 03 into importable, reusable
functions. Exposes build, train, evaluate, and serialization helpers
so that the predictor and tests can share a single source of truth.

Run directly to retrain and re-serialize the model:
    uv run python -m src.ml_pipeline
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    FEATURE_TYPES,
    MODEL_PATH,
    ORDINAL_ORDERS,
    SELECTED_FEATURES,
    STATS_PATH,
)

logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42

# All candidate models evaluated during training selection.
_CANDIDATE_MODELS: dict[str, Any] = {
    "Ridge": Ridge(alpha=10.0),
    "Lasso": Lasso(alpha=0.001, max_iter=5000),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}


def build_preprocessor() -> ColumnTransformer:
    """Build a fresh, unfitted ColumnTransformer for the selected features.

    Sub-pipelines:
    - numeric  : SimpleImputer(median)       → StandardScaler
    - ordinal  : SimpleImputer(most_frequent) → OrdinalEncoder(explicit order)
    - nominal  : SimpleImputer(most_frequent) → OneHotEncoder(ignore unknown)

    Returns:
        Unfitted ColumnTransformer ready to be embedded in a Pipeline.
    """
    numeric_features = [f for f, t in FEATURE_TYPES.items() if t == "numeric"]
    ordinal_features = [f for f, t in FEATURE_TYPES.items() if t == "ordinal"]
    nominal_features = [f for f, t in FEATURE_TYPES.items() if t == "nominal"]

    ordinal_categories = [
        ORDINAL_ORDERS.get(col, ["None", "Po", "Fa", "TA", "Gd", "Ex"])
        for col in ordinal_features
    ]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    nominal_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("nom", nominal_transformer, nominal_features),
        ],
        remainder="drop",
    )


def build_pipeline(model: Any) -> Pipeline:
    """Wrap a preprocessor and estimator into a single sklearn Pipeline.

    The returned Pipeline accepts raw feature DataFrames with string
    categoricals and unscaled numerics — no pre-processing required
    by the caller.

    Args:
        model: Unfitted sklearn estimator (e.g. GradientBoostingRegressor).

    Returns:
        Unfitted Pipeline(preprocessor, model).
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[str, Any]:
    """Run 5-fold CV on all candidate models; return the best by CV RMSE.

    Each candidate is wrapped with the standard preprocessor. The target
    is log1p-transformed before scoring so RMSE is on the log scale
    (consistent with the training objective).

    Args:
        X_train: Training feature DataFrame (selected columns only).
        y_train: Training SalePrice series (dollar scale).

    Returns:
        Tuple of (model_name, unfitted_estimator) for the winner.
    """
    y_log = np.log1p(y_train)
    best_name, best_estimator, best_score = "", None, float("inf")
    results: list[tuple[str, float, float]] = []

    for name, estimator in _CANDIDATE_MODELS.items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(
            pipeline, X_train, y_log, cv=5,
            scoring="neg_root_mean_squared_error",
        )
        mean_cv = float(-scores.mean())
        std_cv = float(scores.std())
        results.append((name, mean_cv, std_cv))
        logger.info("CV RMSE  %-20s %.4f ± %.4f", name, mean_cv, std_cv)
        if mean_cv < best_score:
            best_name, best_estimator, best_score = name, estimator, mean_cv

    logger.info("Winner: %s (CV RMSE %.4f)", best_name, best_score)
    return best_name, best_estimator


def _compute_metrics(
    y_true: pd.Series, y_pred: "np.ndarray[Any, np.dtype[np.floating[Any]]]"
) -> dict[str, float]:
    """Compute RMSE, MAE, and R² in original dollar scale.

    Args:
        y_true: True SalePrice values (dollars, not log-transformed).
        y_pred: Predicted SalePrice values (dollars).

    Returns:
        Dict with keys: rmse, mae, r2.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, dict[str, float]]:
    """Fit a pipeline on training data and report metrics on train + val.

    Target is log1p-transformed internally; all returned metrics are in
    original dollar scale (expm1 back-transform applied before scoring).

    Args:
        pipeline: Unfitted sklearn Pipeline (will be mutated in place).
        X_train: Training feature DataFrame (selected columns only).
        y_train: Training SalePrice series.
        X_val: Validation feature DataFrame.
        y_val: Validation SalePrice series.

    Returns:
        {"train": {rmse, mae, r2}, "val": {rmse, mae, r2}}
    """
    y_train_log = np.log1p(y_train)

    logger.info("Fitting pipeline on %d training samples", len(X_train))
    pipeline.fit(X_train, y_train_log)

    train_pred = np.expm1(pipeline.predict(X_train))
    val_pred = np.expm1(pipeline.predict(X_val))

    train_m = _compute_metrics(y_train, train_pred)
    val_m = _compute_metrics(y_val, val_pred)

    logger.info(
        "Train RMSE=$%.0f R²=%.4f | Val RMSE=$%.0f R²=%.4f",
        train_m["rmse"],
        train_m["r2"],
        val_m["rmse"],
        val_m["r2"],
    )
    return {"train": train_m, "val": val_m}


def compute_training_stats(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the training-stats dict that predictor and Stage 2 consume.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training SalePrice series.
        extra: Optional additional keys to merge (e.g. cv_metrics).

    Returns:
        Dict with sale_price_stats and any extra keys supplied.
    """
    stats: dict[str, Any] = {
        "selected_features": SELECTED_FEATURES,
        "feature_types": {
            "numeric": [f for f, t in FEATURE_TYPES.items() if t == "numeric"],
            "ordinal": [f for f, t in FEATURE_TYPES.items() if t == "ordinal"],
            "nominal": [f for f, t in FEATURE_TYPES.items() if t == "nominal"],
        },
        "target_transform": "log1p",
        "train_size": len(X_train),
        "sale_price_stats": {
            "mean": float(y_train.mean()),
            "median": float(y_train.median()),
            "std": float(y_train.std()),
            "min": float(y_train.min()),
            "max": float(y_train.max()),
            "q25": float(y_train.quantile(0.25)),
            "q75": float(y_train.quantile(0.75)),
        },
    }
    if extra:
        stats.update(extra)
    return stats


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    """Serialize a fitted Pipeline to a .joblib file.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path: Destination path (defaults to models/best_model_v1.joblib).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    size_kb = path.stat().st_size / 1024
    logger.info("Model saved: %s (%.1f KB)", path, size_kb)


def save_training_stats(
    stats: dict[str, Any],
    path: Path = STATS_PATH,
) -> None:
    """Write training statistics to a JSON file.

    Args:
        stats: Training metadata and metrics dict.
        path: Destination path (defaults to models/training_stats.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Training stats saved: %s", path)


# ---------------------------------------------------------------------------
# Script entry-point: retrain from processed data and re-serialize
# ---------------------------------------------------------------------------


def _load_processed_splits() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Load the six processed CSV files from data/processed/.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises:
        FileNotFoundError: If any processed CSV is missing.
    """

    def _load_X(name: str) -> pd.DataFrame:
        path = DATA_PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {path}. Run Notebook 02 first."
            )
        return pd.read_csv(path, index_col=0)

    def _load_y(name: str) -> pd.Series:
        path = DATA_PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {path}. Run Notebook 02 first."
            )
        return pd.read_csv(path, index_col=0).squeeze()

    return (
        _load_X("X_train"),
        _load_X("X_val"),
        _load_X("X_test"),
        _load_y("y_train"),
        _load_y("y_val"),
        _load_y("y_test"),
    )


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # Load raw CSV — the ColumnTransformer handles imputation, ordinal encoding,
    # and OHE internally, so we need raw string categoricals (e.g. Neighborhood)
    # not the pre-encoded output from Notebook 02.
    df = pd.read_csv(DATA_RAW_PATH)
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("/", "", regex=False)
    )
    df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
    df = df.dropna(subset=["SalePrice"])

    # Engineer derived features used by SELECTED_FEATURES
    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0).fillna(0)
        + df.get("1stFlrSF", 0).fillna(0)
        + df.get("2ndFlrSF", 0).fillna(0)
    )
    df["TotalBath"] = (
        df.get("FullBath", 0).fillna(0)
        + 0.5 * df.get("HalfBath", 0).fillna(0)
        + df.get("BsmtFullBath", 0).fillna(0)
        + 0.5 * df.get("BsmtHalfBath", 0).fillna(0)
    )

    X = df[SELECTED_FEATURES]
    y = df["SalePrice"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 × 0.80 = 0.20
    )
    logger.info(
        "Loaded raw CSV — train:%d  val:%d  test:%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    X_train_sel = X_train
    X_val_sel = X_val
    X_test_sel = X_test

    # Select best model via 5-fold CV across all 4 candidates
    best_name, best_estimator = select_best_model(X_train_sel, y_train)

    # Train winner on full training data and evaluate
    best_pipeline = build_pipeline(best_estimator)
    metrics = train_and_evaluate(
        best_pipeline, X_train_sel, y_train, X_val_sel, y_val
    )

    # Evaluate on test set exactly once
    test_pred = np.expm1(best_pipeline.predict(X_test_sel))
    test_metrics = _compute_metrics(y_test, test_pred)
    logger.info(
        "Test  RMSE=$%.0f R²=%.4f",
        test_metrics["rmse"],
        test_metrics["r2"],
    )

    stats = compute_training_stats(
        X_train_sel,
        y_train,
        extra={
            "model_name": best_name,
            "val_metrics": metrics["val"],
            "test_metrics": test_metrics,
        },
    )

    save_model(best_pipeline)
    save_training_stats(stats)
    logger.info("Done. %s saved as best model.", best_name)
