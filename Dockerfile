# Hugging Face Spaces (Docker SDK) image for RepoSage.
FROM python:3.12-slim

# Caches + the on-disk Qdrant store live under /app/.cache and /app/qdrant_data
# so we can PRE-BUILD them at image-build time (below) — that makes the first
# user query fast instead of paying a cold-start model-download + index.
ENV HOME=/tmp \
    HF_HOME=/app/.cache/hf \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    REPOSAGE_QDRANT_PATH=/app/qdrant_data \
    REPOSAGE_INDEX_REPO=/app/reposage \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY reposage ./reposage

# Pre-download the embedding model AND build the index at BUILD time. Tolerant of
# failure: if it can't (e.g. no network during build), the app lazy-indexes at
# runtime instead — slower first query, but still works.
RUN python -c "from reposage.ingest import ingest_repo; from reposage.index import index_chunks; print('prebuilt index:', index_chunks(ingest_repo('/app/reposage')), 'chunks')" \
    || echo "pre-index skipped; will index at runtime"
# make caches + index readable/writable by the Spaces runtime user (uid 1000)
RUN chmod -R a+rwX /app/.cache /app/qdrant_data 2>/dev/null || true

EXPOSE 7860
CMD ["uvicorn", "reposage.web:app", "--host", "0.0.0.0", "--port", "7860"]
