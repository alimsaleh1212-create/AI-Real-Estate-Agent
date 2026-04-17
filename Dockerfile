# syntax=docker/dockerfile:1
# ── Base ─────────────────────────────────────────────────────────────────────
# python:3.11-slim — minimal Debian image (~150 MB vs ~900 MB full).
FROM python:3.11-slim

WORKDIR /app

# Disable .pyc generation; force stdout/stderr flushing for log visibility.
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── uv ───────────────────────────────────────────────────────────────────────
# Copy the uv binary from the official image — pinned for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

# ── Dependencies (cached layer) ───────────────────────────────────────────────
# Copy manifests first so Docker reuses this layer until they change.
COPY pyproject.toml uv.lock ./

# Install production deps from the lockfile exactly — no floating versions.
# --frozen: treat uv.lock as authoritative, fail if it is out of date.
# --no-dev: exclude test/lint/notebook tooling from the image.
# --no-install-project: the project itself is plain source, not a package.
RUN uv sync --frozen --no-dev --no-install-project

# Add the venv to PATH so uvicorn / streamlit are callable as plain commands.
ENV PATH="/app/.venv/bin:$PATH"

# ── Application code ──────────────────────────────────────────────────────────
COPY src/         ./src/
COPY app/         ./app/
COPY ui/          ./ui/
COPY .streamlit/  ./.streamlit/

# ── Artifacts ─────────────────────────────────────────────────────────────────
# Pre-trained model — baked into the image so the container is self-contained.
# Retrain → rebuild the image to embed the new model.
COPY models/      ./models/

# Raw dataset — required by the Streamlit analytics dashboard at runtime.
COPY data/raw/    ./data/raw/

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8000: FastAPI (uvicorn) | 8501: Streamlit
# EXPOSE is documentation only; port binding is defined in docker-compose.
EXPOSE 8000 8501

# ── Default command ───────────────────────────────────────────────────────────
# Runs FastAPI. The Streamlit service overrides this in docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
