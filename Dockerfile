# Hugging Face Spaces (Docker SDK) image for RepoSage.
FROM python:3.12-slim

# /tmp is always writable in the Space container — point caches + the on-disk
# Qdrant store there, and index the app's own code on first request.
ENV HOME=/tmp \
    REPOSAGE_QDRANT_PATH=/tmp/qdrant \
    REPOSAGE_INDEX_REPO=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY reposage ./reposage

EXPOSE 7860
CMD ["uvicorn", "reposage.web:app", "--host", "0.0.0.0", "--port", "7860"]
