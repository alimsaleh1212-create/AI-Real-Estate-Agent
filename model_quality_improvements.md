# Plan: Notebook 03 — Model Quality Improvements

## Context
Post-training evaluation revealed three concrete weaknesses:
1. **Overfitting**: Train R²=0.979 vs Test R²=0.905 (gap=0.074, borderline)
2. **Multicollinearity**: GrLivArea ↔ TotalSF at r=0.866 (GrLivArea is a direct component of TotalSF)
3. **Unreliable model selection**: Fixed val RMSE=$41k vs Test RMSE=$25k (64% gap from unlucky split)

User also asked about: scaling/encoding correctness, KNN imputation vs median.

All changes go in **Notebook 03 only** — no new files, no changes to Notebooks 01/02 or src/.

---

## Answers to User's Questions

### Scaling & Encoding — Already Correct for GBR
- **Numeric** → `SimpleImputer(median)` → `StandardScaler` ✓
- **Ordinal** (KitchenQual, ExterQual) → `SimpleImputer(mode)` → `OrdinalEncoder(explicit_order)` — no scaler after, which is **correct for tree-based models** (GBR doesn't use distances)
- **Nominal** (Neighborhood) → `SimpleImputer(mode)` → `OneHotEncoder` ✓
- Note: Ridge would benefit from scaling ordinal features, but GBR won selection so this doesn't affect the final model.

### KNN Imputation — Not Worth Adding
All 7 selected numeric features have **zero nulls** in the processed data. TotalBsmtSF's zeros are structural (no basement), not missing — correctly filled with 0. LotFrontage (the only feature with genuine missingness) is **not in the selected feature set**. `SimpleImputer(median)` stays as a defensive no-op for inference safety. Adding KNNImputer would add latency with zero practical benefit.

---

## Changes (all in Notebook 03)

### Change 1 — Add imports (Cell 1)
Add to bottom of import block:
```python
from sklearn.model_selection import cross_val_score, KFold
```
(No LinearRegression import needed — VIF uses `.score()` directly)

### Change 2 — Replace GrLivArea with BsmtQual (Cell 16)
**Why BsmtQual over other candidates:**
| Candidate | r(target) | r(nearest neighbor) | Decision |
|-----------|-----------|---------------------|----------|
| GarageArea | 0.637 | 0.891 vs GarageCars | ✗ moves problem |
| TotRmsAbvGrd | 0.488 | 0.670 vs TotalSF | ✓ but weak target r |
| **BsmtQual** | **0.670** | **0.444 vs TotalBsmtSF** | **✓ best choice** |
| OverallCond | 0.055 | — | ✗ too weak |

Replace in `selected_features` list:
```python
selected_features = [
    "OverallQual",    # Numeric  — #1 predictor (1–10 integer scale)
    "TotalSF",        # Numeric  — engineered total sqft (basement + floors)
    "GarageCars",     # Numeric  — garage capacity
    "TotalBath",      # Numeric  — engineered weighted bathroom count
    "YearBuilt",      # Numeric  — age proxy
    "TotalBsmtSF",    # Numeric  — basement sqft
    "KitchenQual",    # Ordinal  — top Spearman ordinal
    "BsmtQual",       # Ordinal  — replaces GrLivArea; r=0.67 with target,
                      #            r=0.44 with TotalBsmtSF (VIF-safe)
    "Neighborhood",   # Nominal  — top ANOVA F-stat
    "ExterQual",      # Ordinal  — strong Spearman
]
# GrLivArea removed: it equals (1stFlrSF + 2ndFlrSF) which is inside TotalSF.
# This caused r=0.866 with TotalSF and effectively duplicated information.
```
Also update `selected_ordinal` assignment in Cell 22 to include `BsmtQual`:
```python
selected_ordinal = [f for f in selected_features if f in ordinal_features]
# ordinal_features already contains BsmtQual from feature_metadata
```
And add BsmtQual's ordinal order to `ordinal_orders` dict (Cell 22 or wherever it's defined):
```python
ordinal_orders["BsmtQual"] = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
```

### Change 3 — Add VIF computation cell (after Cell 20, the heatmap cell)
Insert new cell:
```python
# VIF: quantifies collinearity beyond pairwise r. VIF > 10 = action needed.
# VIF_i = 1 / (1 - R²) from regressing feature_i on all other numeric features.
from sklearn.linear_model import LinearRegression

def compute_vif(df: pd.DataFrame) -> pd.Series:
    cols = list(df.columns)
    vif_vals = {}
    for col in cols:
        others = [c for c in cols if c != col]
        r2 = LinearRegression().fit(
            df[others].values, df[col].values
        ).score(df[others].values, df[col].values)
        vif_vals[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return pd.Series(vif_vals, name="VIF").sort_values(ascending=False)

sel_numeric_check = [f for f in selected_features if f in numeric_features]
vif = compute_vif(X_train[sel_numeric_check].dropna())
print("VIF for selected numeric features (threshold: 10):")
for feat, val in vif.items():
    flag = "  *** ACTION" if val > 10 else ("  * monitor" if val > 5 else "")
    print(f"  {feat:18s}: VIF = {val:6.2f}{flag}")
```
Expected output: all VIFs < 10 (was TotalSF=313, GrLivArea=127 before fix).

Also update the `high_corr` message in Cell 20:
```python
# change the comment from "→ GBR handles multicollinearity; Ridge may downweight both."
# to:
# "→ See VIF cell below. GrLivArea removed and replaced with BsmtQual."
```

### Change 4 — GBR hyperparameters (Cell 25)
Three regularization changes:
```python
gb_pipeline = Pipeline(steps=[
    ("preprocessor", build_preprocessor()),
    ("model", GradientBoostingRegressor(
        n_estimators=300,          # upper cap; early stopping reduces actual count
        max_depth=3,               # was 4; shallower trees reduce memorization
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,        # NEW: prevents single-house leaf splits
        n_iter_no_change=20,       # NEW: early stopping trigger
        validation_fraction=0.1,   # NEW: 10% of train held for early stopping
        random_state=RANDOM_STATE,
    )),
])
gb_pipeline.fit(X_train_sel, y_train_log)
actual_trees = gb_pipeline.named_steps["model"].n_estimators_
print(f"GBR used {actual_trees} trees (early stopping from cap of 300)")
```
Expected: `actual_trees` ≈ 110–150. Train R² drops to ~0.92–0.93, gap narrows to ~0.02.

### Change 5 — Add 5-fold CV block (new cell after GBR fit, before bar chart)
```python
CV_FOLDS = 5
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

ridge_cv = -cross_val_score(ridge_pipeline, X_train_sel, y_train_log,
                             cv=kf, scoring="neg_root_mean_squared_error")
gb_cv    = -cross_val_score(gb_pipeline,    X_train_sel, y_train_log,
                             cv=kf, scoring="neg_root_mean_squared_error")

print("5-Fold CV RMSE (log scale) — primary model selection metric")
print(f"  Ridge:            {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")
print(f"  GradientBoosting: {gb_cv.mean():.4f} ± {gb_cv.std():.4f}")
print(f"\n(Val RMSE is secondary — fixed split was unreliable: $41k vs test $25k)")
```

### Change 6 — Update model selection (Cell 29)
Replace val-RMSE criterion with CV-RMSE criterion:
```python
# PRIMARY selection: CV RMSE (robust to split luck)
if ridge_cv.mean() < gb_cv.mean():
    best_pipeline, best_name, best_val_m, best_cv = ridge_pipeline, "Ridge", ridge_val_m, ridge_cv
else:
    best_pipeline, best_name, best_val_m, best_cv = gb_pipeline, "GradientBoosting", gb_val_m, gb_cv

print(f"Best model: {best_name}")
print(f"  CV RMSE:  {best_cv.mean():.4f} ± {best_cv.std():.4f} (primary)")
print(f"  Val RMSE: ${best_val_m['rmse']:,.0f} (secondary)")
```

### Change 7 — Add cv_metrics to training_stats.json (Cell 34)
Add to `training_stats` dict before `json.dump`:
```python
"cv_metrics": {
    "folds": CV_FOLDS,
    "ridge_cv_rmse_mean": float(ridge_cv.mean()),
    "ridge_cv_rmse_std":  float(ridge_cv.std()),
    "gbr_cv_rmse_mean":   float(gb_cv.mean()),
    "gbr_cv_rmse_std":    float(gb_cv.std()),
},
"multicollinearity_fix": {
    "removed": "GrLivArea",
    "reason": "component of TotalSF; r=0.866 pairwise",
    "added": "BsmtQual",
    "added_target_r": 0.67,
},
```

---

## Execution Order
```
Cell 1  → add KFold, cross_val_score imports
Cell 16 → swap GrLivArea → BsmtQual in selected_features
Cell 20 → update high_corr comment
NEW VIF → insert after Cell 20
Cell 22 → add ordinal_orders["BsmtQual"]; selected_ordinal auto-picks BsmtQual
Cell 25 → max_depth=3, min_samples_leaf=5, n_iter_no_change=20
NEW CV  → insert after Cell 25 fit; produces ridge_cv, gb_cv
Cell 27 → no change (bar chart still works)
Cell 29 → swap criterion to CV RMSE
Cell 34 → add cv_metrics, multicollinearity_fix to stats JSON
```

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Train R² | 0.979 | ~0.930 |
| Val R² | 0.747 | ~0.840 |
| Test R² | 0.905 | ~0.907 (stable) |
| Train-Test R² gap | 0.074 | ~0.023 |
| Max VIF | 313 | < 10 |
| GBR trees used | 300 | ~120 (early stopping) |
| Model selection criterion | Fixed val RMSE | 5-fold CV RMSE |

---

## Critical Files
- `notebooks/03_feature_selection_and_training.ipynb` — all changes here
- `models/training_stats.json` — regenerated by running the notebook
- `models/best_model_v1.joblib` — regenerated (new feature set + hyperparams)
- `data/processed/feature_metadata.json` — read-only input, no changes needed

## Verification
1. Run all cells top-to-bottom in Notebook 03
2. VIF cell: all numeric features < 10
3. GBR fit: `n_estimators_` prints ~110–150
4. CV cell: GBR CV RMSE < Ridge CV RMSE (confirms model selection)
5. Test evaluation: Test R² ≥ 0.90, Train-Test gap ≤ 0.05
6. `models/training_stats.json` contains `cv_metrics` key
