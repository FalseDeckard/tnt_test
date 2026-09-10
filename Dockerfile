FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

ENV PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

COPY . /app

RUN chmod +x /app/download_data.sh

RUN ./download_data.sh

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('deepvk/USER-bge-m3')"

RUN chown -R app:app /app/data "$HF_HOME"

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1:8000/ || exit 1

CMD ["python", "-m", "app.main"]
