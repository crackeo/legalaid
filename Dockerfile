FROM python:3.12-slim

# tesseract: OCR fallback for scanned law PDFs during `ingest.pipeline update`
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/legalaid

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytesseract pillow

COPY ingest/ ingest/
COPY rag/ rag/
COPY app/ app/
COPY web/ web/
COPY eval/ eval/

# The corpus (raw PDFs, chunks.jsonl, index.db, sessions.db) lives on a
# volume so it survives image upgrades: mount ./corpus at /srv/legalaid/corpus.
VOLUME ["/srv/legalaid/corpus"]

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
