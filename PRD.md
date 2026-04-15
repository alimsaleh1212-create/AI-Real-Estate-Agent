# PRD: AI Real Estate Agent

## Context

Week 2 AIE Bootcamp project. Deadline: Thursday 2026-04-16 midnight. Presentation: Friday 2026-04-17.

**What we're building:** An AI real estate agent — natural-language property description in, structured feature extraction (LLM Stage 1), ML price prediction, contextual interpretation (LLM Stage 2) — served from a Dockerized FastAPI app with a Streamlit UI.

| Key Decision | Choice | Rationale |
|-------------|--------|-----------|
| Dataset | Ames Housing (2,930 rows, 80+ features) | Recommended by brief, rich feature set |
| Feature selection | Data-driven (statistical tests) → 10 features | Not hand-picked; justified with Pearson, MI, ANOVA, Spearman |
| LLM provider | Google Gemini API (free tier, `gemini-2.5-flash`) | Zero cost — 15 RPM, 1M tokens/day |
| Env management | `uv` (fast Python package manager + venv) | Modern, fast, lockfile support |
| ML models | Ridge Regression + GradientBoostingRegressor | Linear baseline + non-linear; two swappable as required by brief |
| UI | Streamlit | Simplest, fewest deps, good for forms |
| Bonus | Yes — market insights with intent classification | Strengthens presentation |

---

## Architecture — The Full Chain

```
User types: "3-bed ranch, big garage, good neighborhood"
        |
        v
+---------------------------------+
|  Intent Classifier (Gemini)     |  "prediction" or "analysis"?
|  (BONUS)                        |  Routes to correct pipeline
+-------+---------------+---------+
        | prediction    | analysis
        v               v
+-------------+  +--------------------------+
| PREDICTION  |  | MARKET INSIGHTS (BONUS)  |
| PIPELINE    |  | Pre-computed stats ->     |
| (below)     |  | Gemini -> narration       |
+-------------+  +--------------------------+

--- Prediction Pipeline ---

User query
    |
    v
+---------------------------------+
|  LLM Stage 1 (Gemini API)      |  Extract features -> Pydantic schema
|  - Maps NL -> typed values      |  with completeness metadata
|  - Reports missing features     |  (which extracted, which unknown)
+---------+-----------------------+
          | ExtractedFeatures (Pydantic)
          v
+---------------------------------+
|  UI: Review & Fill Gaps         |  User sees what was extracted,
|                                 |  fills in missing features
+---------+-----------------------+
          | Complete feature dict
          v
+---------------------------------+
|  ML Model (.joblib)             |  scikit-learn Pipeline predicts
|  Loaded at FastAPI startup      |  SalePrice
+---------+-----------------------+
          | predicted_price (float)
          v
+---------------------------------+
|  LLM Stage 2 (Gemini API)      |  Receives: features + prediction
|  - Training data stats          |  + summary stats (median, range)
|  - Contextual interpretation    |  Outputs: narrative explanation
+---------+-----------------------+
          | PredictionResponse (Pydantic)
          v
+---------------------------------+
|  Streamlit UI                   |  Shows prediction + interpretation
+---------------------------------+
```

**What connects Stage 1 to Stage 2:** The `ExtractedFeatures` Pydantic object is the glue. Stage 1 produces it. The ML model consumes it (after gap-filling). The prediction result + those same features + pre-computed training stats are all packed into Stage 2's prompt.

---

## Feature Selection Strategy (Data-Driven)

The brief says "pick 8-12 key features." We don't guess — we run statistical tests on all 80+ Ames columns and select the top 10 based on evidence. Each selected feature gets a written interpretation of why it matters.

### Statistical Tests

| Test | Applies To | What It Measures |
|------|-----------|-----------------|
| Pearson correlation | Numeric vs. SalePrice | Linear relationship strength (|r| > 0.5 = strong) |
| Mutual Information | All features vs. SalePrice | Non-linear dependencies Pearson misses |
| ANOVA F-test | Categorical vs. SalePrice | Whether mean price differs significantly across groups |
| Spearman rank | Ordinal vs. SalePrice | Monotonic relationship for ordered categories |

### Selection Process

1. Rank all features by Pearson |r|, MI score, ANOVA F-stat, Spearman rho
2. Take the union of top-15 from each method
3. Apply constraints:
   - At least 1 nominal categorical (brief requirement)
   - At least 1 ordinal categorical (brief requirement)
   - At least 1 column with missing values (brief requirement)
   - All must be user-describable in natural language
4. Cut to final 10 with written justification per feature

### Expected Winners (hypothesis, confirmed by data)

| Likely Feature | Why We Expect It | Test That Should Surface It |
|---------------|-------------------|---------------------------|
| OverallQual | Quality dominates price in housing literature | Pearson r > 0.7, highest MI |
| GrLivArea | Size is the universal price driver | Pearson r > 0.7 |
| GarageArea/GarageCars | Strong proxy for home size and quality | Pearson r > 0.6 |
| TotalBsmtSF | Adds livable space, has missing values | Pearson r > 0.6, satisfies null req |
| FullBath | More baths = higher price | Moderate r, high MI |
| YearBuilt/YearRemodAdd | Age/condition proxy | Pearson r > 0.5 |
| Neighborhood | Location is everything in real estate | ANOVA F-stat, highest group variance |
| KitchenQual | Ordinal quality indicator | Spearman rank, satisfies ordinal req |
| LotArea | Property size | Moderate r |
| Fireplaces/TotRmsAbvGrd | Amenity/size signal | MI score |

**We do NOT hardcode these.** The notebook runs the tests, prints the rankings, and the final selection is made from the results.

---

## Project Structure

```
project2_ai_real_estate_agent/
├── data/
│   └── raw/                      # Ames CSV (read-only, gitignored)
├── models/                       # Serialized .joblib + stats JSON (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Settings, env vars, constants
│   ├── schemas.py                # All Pydantic models
│   ├── ml_pipeline.py            # Train, evaluate, serialize model
│   ├── predictor.py              # Load model + predict at runtime
│   ├── prompts.py                # Prompt templates (v1, v2) — prompts are code
│   └── llm_chain.py              # Stage 1 + Stage 2 + intent classifier (Gemini)
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   └── ui.py                     # Streamlit UI
├── tests/
│   ├── test_schemas.py
│   ├── test_predictor.py
│   ├── test_llm_chain.py
│   └── test_main.py
├── notebooks/
│   └── eda_and_pipeline.ipynb    # EDA -> Cleaning -> Feature Eng -> Selection -> Training
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── pyproject.toml                # uv project config + pinned deps
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
└── CLAUDE.md
```

---

## Build Blocks

### Block 1: Project Scaffolding

**Goal:** Repo init, `uv` venv, directory structure, deps, dataset download, pre-commit hooks.

**Steps:**
1. `git init` + create `main` branch
2. Create all directories
3. `uv init` + write `pyproject.toml` with pinned deps (separate prod/dev groups)
4. `uv sync` to create `.venv` and install everything
5. Write `.gitignore` (per AIE guidelines template)
6. Write `.dockerignore` (per AIE guidelines template)
7. Write `.env.example` (`GOOGLE_API_KEY=your-gemini-api-key-here`)
8. Write `.pre-commit-config.yaml` (black -> isort -> flake8 -> mypy -> pytest -> gitleaks)
9. Write `src/config.py` — env loading, constants
10. Download Ames Housing CSV into `data/raw/`
11. Empty `__init__.py` files

**Verify:** `uv sync` succeeds. `data/raw/train.csv` exists.

---

### Block 2: Notebook — EDA, Cleaning, Feature Engineering, Feature Selection, Training

**Requirement mapping:** Brief items 01-05. The notebook is a deliverable.

#### 2a. Data Loading & Exploration
- Load all 80+ columns from `data/raw/train.csv`
- `df.shape`, `df.dtypes`, `df.describe()`
- Target variable analysis: SalePrice distribution, skewness check
- Missing value audit: heatmap, percentage per column
- Data types: classify numeric vs. categorical vs. ordinal

#### 2b. Data Cleaning
- Drop columns with >40% nulls (PoolQC, MiscFeature, Alley, Fence) — document each
- Handle remaining nulls on training set only:
  - Numeric: median imputation (robust to outliers)
  - Categorical: mode or "None" category (e.g., GarageType NaN = no garage)
- Outlier detection: box plots, document decisions
- Type corrections: ensure ordinal features are recognized

#### 2c. Feature Engineering
- Derived features to test:
  - `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
  - `HouseAge = YrSold - YearBuilt`
  - `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
  - `HasRemodel = 1 if YearRemodAdd != YearBuilt else 0`
- Log-transform SalePrice if right-skewed
- Encode ordinal features: KitchenQual, ExterQual -> Po=1, Fa=2, TA=3, Gd=4, Ex=5

#### 2d. Feature Selection (Statistical Tests)
- Pearson correlation for numeric features
- Mutual Information for all features (captures non-linear)
- ANOVA F-test for categorical features
- Spearman rank for ordinal features
- Combined ranking table with all scores
- Final 10 selected with written justification per feature
- Constraints verified: 1+ nominal, 1+ ordinal, 1+ with nulls, all user-describable

#### 2e. Three-Way Split (Brief item 01)
- 70% train / 15% validation / 15% test
- No leakage: all transformers `.fit()` on X_train only

#### 2f. Preprocessing — ColumnTransformer (Brief items 02, 03)

| Column Group | Pipeline Steps | Justification |
|-------------|---------------|---------------|
| Numeric | `SimpleImputer(median)` -> `StandardScaler` | Median robust to outliers. Scaling needed for Ridge. |
| Ordinal | `SimpleImputer(most_frequent)` -> `OrdinalEncoder(categories=[[ordered]])` | Preserves ranking (Po < Fa < TA < Gd < Ex). |
| Nominal | `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")` | No natural order. `handle_unknown` prevents inference errors. |

#### 2g. Model Training & Evaluation (Brief items 04, 05)
- Two swappable models: Ridge Regression + GradientBoostingRegressor
- Both wrapped in `Pipeline` with the `ColumnTransformer` preprocessor
- Metrics: RMSE, MAE, R-squared on train + validation
- Pick best by val RMSE
- Run best on test set exactly once
- Serialize: `joblib.dump(best_pipeline, "models/best_model_v1.joblib")`
- Save training stats to `models/training_stats.json`

**Verify:** Both models train. Val RMSE reasonable. `.joblib` and `.json` saved.

---

### Block 3: ML Pipeline Module (`src/ml_pipeline.py`)

Extracts reusable training logic from notebook into importable code.

Functions:
- `build_preprocessor()` -> `ColumnTransformer`
- `build_pipeline(preprocessor, model)` -> `Pipeline`
- `train_and_evaluate(pipeline, X_train, y_train, X_val, y_val)` -> metrics dict
- `compute_training_stats(X_train, y_train)` -> dict
- `save_model(pipeline, path)` -> writes `.joblib`
- `save_training_stats(stats, path)` -> writes `.json`

**Verify:** `uv run python -m src.ml_pipeline` trains and saves artifacts.

---

### Block 4: Pydantic Schemas (`src/schemas.py`)

**Requirement mapping:** Brief item 09 — "Two Pydantic schemas minimum."

**Schema 1 — `ExtractedFeatures`:**
- Each selected feature as `Optional[type]` (None = not mentioned by user)
- `confidence: dict[str, FeatureConfidence]` — maps each feature to "extracted" or "missing"
- Computed properties: `missing_features`, `extracted_features`, `is_complete`

**Schema 2 — `PredictionResponse`:**
- `query`, `extracted_features`, `predicted_price`, `interpretation`, `error`

**Additional:** `PredictionRequest`, `InsightRequest`, `InsightResponse` (bonus).

**Error handling:** Catch `GoogleAPIError`, `ValidationError`, malformed JSON. Retry once on parse failure. Return fallback with `error` field. Demonstrate one failure case in tests.

Note: Exact fields on `ExtractedFeatures` depend on which 10 features win in Block 2d.

**Verify:** `uv run python -c "from src.schemas import ExtractedFeatures, PredictionResponse"` works.

---

### Block 5: LLM Prompt Chain (`src/prompts.py` + `src/llm_chain.py`)

**Requirement mapping:** Brief items 06, 07, 08.

#### Prompts (in `src/prompts.py` — prompts are code)

| Prompt | Purpose | Key Instruction |
|--------|---------|----------------|
| `EXTRACTION_PROMPT_V1` | Stage 1 direct instruction | List features + types + ranges. "Return JSON, null if not mentioned, never guess." |
| `EXTRACTION_PROMPT_V2` | Stage 1 few-shot | Same + 2-3 example query->JSON pairs |
| `INTERPRETATION_PROMPT` | Stage 2 interpretation | "Compare to median, identify drivers, reference neighborhood stats. 3-4 sentences." |
| `INTENT_PROMPT` | Bonus: classify intent | Returns "prediction" or "analysis" |
| `INSIGHTS_PROMPT` | Bonus: market narration | "Answer using ONLY these stats. Cite numbers." |

#### Functions (in `src/llm_chain.py`)

- `extract_features(query)` -> `ExtractedFeatures`
- `predict_and_interpret(features, price, stats)` -> `str`
- `classify_intent(query)` -> `str` (bonus)
- `generate_market_insights(query, stats)` -> `str` (bonus)

Uses Gemini's `response_mime_type="application/json"` for structured Stage 1 output.

#### Prompt Versioning (item 08 — in notebook)

Test both v1 and v2 on 3+ queries:
1. "3-bedroom ranch with a big garage in a good neighborhood" (partial)
2. "2500 sqft, 4 beds 2 baths, Northridge, excellent kitchen..." (complete)
3. "cheap house" (minimal — stress test)
4. "What's the average price in Northridge Heights?" (bonus: analysis intent)

Log: version, query, extracted count, missing count, schema valid, notes. Pick winner with evidence.

**Verify:** `extract_features("3 bed house")` returns valid `ExtractedFeatures`.

---

### Block 6: Predictor (`src/predictor.py`)

- `load_model()` — `joblib.load` at startup, cached module-level
- `get_stats()` — return training stats for Stage 2 / insights
- `predict_price(features)` — `ExtractedFeatures` -> DataFrame -> `model.predict()` -> float

**Gap handling:** `None` values become `NaN` in DataFrame. Pipeline's `SimpleImputer` fills with training medians. Not "silent" — UI showed what was missing.

**Verify:** `predict_price(features_with_nulls)` returns a positive float.

---

### Block 7: FastAPI App (`app/main.py`)

| Route | Method | Response | Purpose |
|-------|--------|----------|---------|
| `/predict` | POST | `PredictionResponse` | Main chain: extract -> predict -> interpret |
| `/insights` | POST | `InsightResponse` | Bonus: market analysis |
| `/health` | GET | `{"status": "ok"}` | Readiness check |
| `/features` | GET | Feature definitions | UI builds gap-filling form from this |

Startup: load model + stats + validate `GOOGLE_API_KEY`.
Errors: catch exceptions, log them, return response with `error` field — never raw stack traces.

**Verify:** `uv run uvicorn app.main:app` -> `curl localhost:8000/health` -> 200.

---

### Block 8: Streamlit UI (`app/ui.py`)

**Two-step flow** (matches brief: "UI lets user review and fill gaps before prediction"):

1. User enters query -> "Analyze" -> calls Stage 1 only -> shows extracted features + gap form
2. User fills gaps -> "Get Price Prediction" -> predict + Stage 2 -> shows price + interpretation

**When 4/10 extracted:** extracted values highlighted, remaining as input fields, message: "Extracted 4 of 10 features. Fill in more for a better prediction, or proceed with defaults."

**Bonus:** If intent = "analysis", shows market narrative instead of prediction flow.

**Verify:** `uv run streamlit run app/ui.py` -> full flow test in browser.

---

### Block 9: Docker (`Dockerfile` + `.dockerignore` + `docker-compose.yml`)

Dockerfile written from scratch. Every instruction justified (brief rule: "explain every instruction or you don't understand it").

- `FROM python:3.11-slim` — minimal base
- `COPY pyproject.toml` first for layer caching
- `RUN pip install --no-cache-dir .` — production deps only
- `COPY models/` — bakes trained model into image
- `EXPOSE 8000 8501` — documents ports
- `CMD uvicorn` — binds `0.0.0.0` so container is accessible from outside

`docker-compose.yml` runs both `api` (FastAPI) and `ui` (Streamlit) services.

**Verify:** `docker compose up --build` -> both ports accessible from host.

---

### Block 10: Tests (`tests/`)

| Test File | What It Tests |
|-----------|--------------|
| `test_schemas.py` | Pydantic validation, missing features, completeness, edge cases |
| `test_predictor.py` | Model load, full prediction, partial prediction (NaN handling) |
| `test_llm_chain.py` | Mock Gemini, extraction parsing, **failure case: malformed output** |
| `test_main.py` | FastAPI endpoints via TestClient, error responses, health check |

**Demonstrated failure case (item 09):** Mock Gemini returning invalid text -> `extract_features()` catches parse error -> returns error response with no prediction.

**Verify:** `uv run pytest --cov=src --cov-report=term-missing` -> all pass, >80% coverage.

---

### Block 11: Notebook Finalization

- Add Colab setup cell: `!pip install ...`
- Ensure all sections run top-to-bottom
- Prompt experiments section with comparison table
- Clear markdown headers

**Verify:** Open in Colab -> Runtime -> Run All -> completes.

---

## End-to-End Data Flow

```
1. User types: "3-bed ranch, big garage, nice neighborhood"
   |
2. Streamlit -> POST /predict {"query": "3-bed ranch..."}
   |
3. FastAPI -> extract_features() -> Gemini Stage 1:
   |  Returns ExtractedFeatures: bedroom=3, garage=800, overall_qual=7
   |  confidence: {bedroom: "extracted", garage: "extracted", ...rest: "missing"}
   |
4. FastAPI returns ExtractedFeatures to Streamlit
   |  UI: "Extracted 3/10" + gap-filling form
   |
5. User fills: living_area=2000, baths=2 -> sends back
   |
6. predict_price(features):
   |  -> DataFrame (5 NaN) -> Pipeline imputer fills with train medians
   |  -> GBR predicts -> $245,000
   |
7. predict_and_interpret(features, $245k, stats) -> Gemini Stage 2:
   |  "This home's estimated $245,000 is 50% above the Ames median of
   |   $163,000. The high quality rating and large garage drive the price..."
   |
8. UI displays: $245,000 + interpretation
```

---

## Execution Order & Dependencies

```
Block 1  (Scaffold + uv)    -- no deps
  |
Block 2  (Notebook: EDA -> Clean -> Feature Eng -> Select -> Train)  -- needs data
  |
Block 3  (src/ml_pipeline.py)  -- extracts reusable code from Block 2
  |
Block 4  (Schemas)  -- needs feature list confirmed in Block 2d
  |
Block 5  (LLM Chain)  -- needs schemas + GOOGLE_API_KEY
  |
Block 6  (Predictor)  -- needs model .joblib + schemas
  |
Block 7  (FastAPI)  -- needs predictor + chain + schemas
  |
Block 8  (Streamlit UI)  -- needs FastAPI running
  |
Block 9  (Docker)  -- needs everything working locally
  |
Block 10 (Tests)  -- written alongside, finalized here
  |
Block 11 (Notebook finalize)  -- add prompt experiments, polish for Colab
```

---

## Verification Checklist

| # | Check | Command |
|---|-------|---------|
| 1 | uv venv works | `uv sync && uv run python --version` |
| 2 | Dataset exists | `ls data/raw/train.csv` |
| 3 | Notebook runs | All cells execute, feature selection table printed |
| 4 | Model serialized | `ls models/best_model_v1.joblib models/training_stats.json` |
| 5 | Schemas import | `uv run python -c "from src.schemas import ExtractedFeatures"` |
| 6 | LLM chain works | Test call to `extract_features()` returns valid schema |
| 7 | FastAPI starts | `uv run uvicorn app.main:app` -> `curl localhost:8000/health` |
| 8 | POST /predict | `curl -X POST localhost:8000/predict -d '{"query":"3 bed house"}'` |
| 9 | Streamlit loads | `uv run streamlit run app/ui.py` -> full flow |
| 10 | Docker works | `docker compose up --build` -> both ports accessible |
| 11 | Tests pass | `uv run pytest --cov=src --cov-report=term-missing` -> 80%+ |
| 12 | Linting passes | `uv run black --check . && uv run isort --check . && uv run flake8 .` |
| 13 | Notebook Colab | Upload -> Run All -> completes |

---

## Security — Prompt Injection Prevention

### Threat Model

The prediction pipeline embeds user input at 4 points:

1. Stage 1 extraction prompt — user query → `EXTRACTION_PROMPT_V2`
2. Intent classifier — user query → `INTENT_PROMPT`
3. Market insights — user query → `INSIGHTS_PROMPT`
4. Stage 2 secondary injection — Stage 1 string outputs (Neighborhood) → `INTERPRETATION_PROMPT`

Surface 4 is the subtlest: the attacker doesn't break Stage 1's JSON structure, just
crafts a query that causes Stage 1 to return a malicious string in `Neighborhood`, which
is then embedded in Stage 2's prompt.

### Controls

| Control | Where | What It Does |
|---------|-------|--------------|
| `_sanitize_query()` | `llm_chain.py` | Caps at 500 chars, strips control chars, logs suspicious patterns |
| XML delimiters `<user_input>` | `prompts.py` | Structural separation of user content from instructions |
| `_sanitize_feature_string()` | `llm_chain.py` | Strips non-alphanumeric chars from Stage 1 string outputs before Stage 2 |
| `max_output_tokens` | `llm_chain.py` | Caps response length; prevents unbounded exfiltration |
| Pydantic schema validation | `schemas.py` | Rejects malformed Stage 1 JSON; validates ordinal values against explicit enum |
| Rate limiting | `app/main.py` | FastAPI middleware caps requests per IP (Block 7) |

### What We Don't Do

- **No blocklist filtering**: Keyword blocklists are trivially bypassed and create false positives. We log, not block.
- **No output content scanning**: Out of scope for this project; production systems should add LLM output moderation.
- **No `system_instruction` separation**: Gemini's `system_instruction` param requires one model instance per prompt type — the delimiter + sanitization approach provides equivalent defence-in-depth.

---

## AIE Bootcamp Coding Guidelines Compliance

| Section | How We Comply |
|---------|--------------|
| 1. Branch Naming | `feature/<description>`, lowercase, hyphens, 2-4 words |
| 2. PR Guidelines | `[TYPE] description`, template with Summary/Changes/Testing/Checklist |
| 3. Commit Messages | `type(scope): Summary` — conventional commits, imperative, <=72 chars |
| 4. Code Style | Black (88), isort (black profile), flake8 (88), mypy strict, type hints on all sigs |
| 5. Naming | snake_case vars/funcs, PascalCase classes, UPPER constants, verb-prefixed funcs |
| 6. Security | No hardcoded secrets, `.env` + `.gitignore`, `os.getenv()` fail-fast, gitleaks |
| 7. .gitignore/.dockerignore | Both per guideline templates |
| 8. Error Handling | Specific exceptions, logged, user-safe messages |
| 9. Testing | AAA pattern, 80%+ coverage, mock externals, `pytest --cov=src` |
| 10. Logging | `logging.getLogger(__name__)`, structured, never print() |
| 11. Dependency Mgmt | Pinned in `pyproject.toml`, separate dev/prod, `uv` |
| 12. Documentation | Google-style docstrings on all public functions/classes/modules |
| 13. Pre-commit | `black -> isort -> flake8 -> mypy -> pytest -> gitleaks` |
