# --- Base image ---
# python:3.11-slim is a minimal Debian image with only the Python runtime.
# Keeps the final image small (~150 MB vs ~900 MB for the full image).
FROM python:3.11-slim

# Set working directory for all subsequent instructions.
WORKDIR /app

# Ensure /app is always on sys.path — required for `import src` when
# streamlit replaces sys.path[0] with the script's directory.
ENV PYTHONPATH=/app

# --- Dependency layer (cached unless pyproject.toml changes) ---
# Copy only the manifest first so Docker can cache this layer.
# Rebuilding deps only happens when pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./

# Install production dependencies from pyproject.toml.
# --no-cache-dir prevents pip from storing the download cache inside the image.
# We install the project itself later (COPY src/ below), but installing deps
# first leverages Docker layer caching during iterative builds.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir $(python -c \
        "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); \
         print(' '.join(d['project']['dependencies']))")

# --- Application code ---
COPY src/ ./src/
COPY app/ ./app/

# --- Pre-trained model artifacts ---
# The model is baked into the image so the container is self-contained.
# Re-build the image after retraining to update the embedded model.
COPY models/ ./models/

# --- Ports ---
# 8000: FastAPI (uvicorn)
# 8501: Streamlit
# EXPOSE is documentation — actual port binding is done in docker-compose.
EXPOSE 8000 8501

# --- Default command: run FastAPI ---
# Override this in docker-compose for the Streamlit service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
