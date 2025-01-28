FROM python:3.10-slim

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

COPY . /app

COPY download_data.sh /app/download_data.sh
RUN chmod +x /app/download_data.sh

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('deepvk/USER-bge-m3')"

CMD ["/bin/bash", "-c", "/app/download_data.sh && python app/main.py"]
