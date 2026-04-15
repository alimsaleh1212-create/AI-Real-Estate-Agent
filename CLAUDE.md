# CLAUDE.md — AI Real Estate Agent Project

Combined behavioral + technical standards. Follow automatically on every file you create or modify.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that YOUR changes made unused — nothing more.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

---

## 5. Branch Naming

Format: `<type>/<short-description>` — lowercase, hyphens only, 2–4 words.

| Prefix | Use For |
|--------|---------|
| `feature/` | New functionality |
| `bugfix/` | Bug fix |
| `hotfix/` | Urgent production fix |
| `refactor/` | Code restructuring |
| `docs/` | Documentation only |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance / tooling |

Never commit directly to `main` or `develop`.

---

## 6. Commit Messages (Conventional Commits)

Format: `<type>(<scope>): <summary>`

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `security`
- Summary: imperative mood, capitalize first letter, no trailing period, ≤72 chars
- Example: `feat(search): Add price range filter to property query`

---

## 7. Pull Requests

Title: `[TYPE] Short imperative description`

```
## Summary
## Changes
## Testing
## Screenshots (if applicable)
## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed the code
- [ ] Added/updated tests
- [ ] Updated documentation
- [ ] No secrets or credentials in code
- [ ] No linting errors
```

Keep PRs under 400 lines. One concern per PR.

---

## 8. Python Code Style

Toolchain: **Black** (line length 88) · **isort** (`profile = "black"`) · **flake8** (max 88) · **mypy** strict

- 4 spaces, never tabs
- Double quotes for strings
- Trailing commas in all multi-line structures
- Type hints required on every function signature
- 2 blank lines between top-level definitions; 1 between methods

Import order (blank line between each group):
```python
# 1. Standard library
import os
from typing import Optional, List

# 2. Third-party
import pandas as pd

# 3. Local
from src.data_loader import load_dataset
```

---

## 9. Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Variables, functions, modules | `snake_case` | `fetch_listings()` |
| Classes | `PascalCase` | `PropertyFilter` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3` |
| Private attributes | `_leading_underscore` | `self._cache` |
| Boolean vars | reads as question | `is_active`, `has_permission` |
| Collections | plural | `listings`, `price_ranges` |

Functions start with a verb: `get_`, `fetch_`, `load_`, `train_`, `save_`, `validate_`, `process_`, `build_`.
No single-letter names except loop vars (`i`, `j`) or lambdas.

---

## 10. Security — CRITICAL

- **Never** hardcode API keys, tokens, passwords, or connection strings
- Always load secrets via `os.getenv()` with a fail-fast guard:
  ```python
  GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
  if not GOOGLE_API_KEY:
      raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
  ```
- Use `.env` locally — always in `.gitignore`
- Load with `python-dotenv`: `load_dotenv()`
- Never log passwords, tokens, PII, or API keys
- Validate all user input at API boundaries using Pydantic
- Required environment variables for this project: `GOOGLE_API_KEY`

---

## 11. Error Handling

- Never use bare `except:` or `except: pass`
- Always catch specific exception types
- Always log the exception before re-raising
- Never expose stack traces or file paths to end users
- Define custom exception classes for domain errors

---

## 12. Logging

Use `logging`, never `print()` for operational output.

```python
logger = logging.getLogger(__name__)
logger.info("Listings fetched", extra={"count": 42, "city": "Austin"})
logger.error("Prediction failed: %s", e)
```

Levels: `DEBUG` (internals) · `INFO` (milestones) · `WARNING` (recoverable) · `ERROR` (failed op) · `CRITICAL` (service down)

---

## 13. Testing

- Every module → `tests/test_<module>.py`
- Test naming: `test_<behavior>_<condition>`
- Pattern: Arrange – Act – Assert in every test
- Mock all external dependencies (HTTP, filesystem, databases)
- Minimum 80% line coverage; critical paths (API endpoints, model training) 95%+
- Run: `pytest --cov=src --cov-report=term-missing`

---

## 14. Documentation (Google Style)

Every public module, class, and function must have a docstring:

```python
def fetch_listings(city: str, max_price: float) -> list[dict]:
    """Fetch property listings filtered by city and price ceiling.

    Args:
        city: Target city name.
        max_price: Upper price bound in USD.

    Returns:
        List of listing dicts with keys: id, address, price, bedrooms.

    Raises:
        ValueError: If max_price is negative.
    """
```

Inline comments: explain **why**, not **what**.

---

## 15. ML/AI Specific Rules

- Raw data is **read-only** — never modify `data/raw/`
- Processed data → `data/processed/`
- Models saved to `models/` as `<model_name>_v<version>.joblib`
- Always log: dataset shape, feature count, train/test split sizes, all eval metrics
- Notebooks are for exploration only — extract all reusable logic into `src/`
- FastAPI: use Pydantic `BaseModel` for all request/response schemas; separate router files per resource group

---

## 16. Pre-commit Pipeline

```
black . → isort . → flake8 . → mypy . → pytest → gitleaks
```

Configure in `.pre-commit-config.yaml`. Run `uv run pre-commit install` once after cloning.

---

## 17. Dependency Management with `uv`

This project uses `uv` for virtual environment and dependency management. Do **not** use raw `pip` or `venv`.

```bash
uv sync                        # Install all deps (prod + dev) into .venv
uv sync --no-dev               # Install production deps only
uv run python ...              # Run Python inside the venv
uv run pytest                  # Run tests inside the venv
uv run uvicorn app.main:app    # Start FastAPI
uv run streamlit run app/ui.py # Start Streamlit UI
uv add <package>               # Add a new production dependency
uv add --dev <package>         # Add a new dev dependency
```

Dependencies live in `pyproject.toml` with **pinned exact versions** (e.g., `fastapi==0.115.12`):

```toml
[project]
dependencies = [...]           # Production only

[dependency-groups]
dev = [...]                    # pytest, black, mypy, etc. — never in prod image
```

Never use `requirements.txt` — `pyproject.toml` is the single source of truth.

---

## 18. LLM Provider

This project uses the **Google Gemini API** (free tier). Model: `gemini-2.5-flash`.

- Client: `google-generativeai` Python SDK
- Auth: `GOOGLE_API_KEY` environment variable
- Use `response_mime_type="application/json"` with `response_schema` for structured extraction (Stage 1)
- Do **not** use `anthropic` SDK or OpenAI SDK in this project

---

## 20. Use These Commands — don't reimplement manually

When performing the following tasks, invoke the corresponding slash command instead of applying the rules from prose:

| Task | Command |
|------|---------|
| Suggest a branch name | `/aie-branch` |
| Write a commit message | `/aie-commit` |
| Open a pull request | `/aie-pr` |
| Review code for compliance | `/aie-review` |
| Add/fix docstrings | `/aie-docs` |
| Generate tests for a module | `/aie-test` |
| Security audit | `/aie-security` |
| Scaffold a new Python module | `/aie-newmodule` |
| Build an end-to-end ML pipeline | `/ml-pipeline` |
