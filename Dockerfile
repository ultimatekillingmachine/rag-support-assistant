FROM python:3.12-slim

# PYTHONUNBUFFERED keeps logs visible; HF_HUB_DISABLE_SYMLINKS avoids the
# symlink/ONNX issue described in the README (same guard as app/__init__.py).
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    HF_HOME=/models \
    EMBEDDING_MODEL=intfloat/multilingual-e5-large \
    STORAGE_BACKEND=sqlite \
    SQLITE_PATH=/data/app.db

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilt when they change.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts
COPY knowledge_base ./knowledge_base
COPY data/eval ./data/eval

# Index and model cache live in volumes, so a container restart re-uses them:
# re-indexing an unchanged knowledge base takes ~0.1s, and the embedding model
# (~2.25 GB) is not downloaded again.
VOLUME ["/data", "/models"]

EXPOSE 8000

# Refresh the index, then serve. Ingest is incremental by design, so this is
# safe to run on every start.
CMD ["sh", "-c", "python -m scripts.ingest_kb && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
