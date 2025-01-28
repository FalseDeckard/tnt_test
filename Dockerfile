FROM python:3.10-slim

RUN apt-get update && apt-get install -y wget dos2unix && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

COPY . /app

RUN dos2unix /app/download_data.sh && chmod +x /app/download_data.sh

RUN ./download_data.sh

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('deepvk/USER-bge-m3')"

CMD ["python", "-m", "app.main"]
