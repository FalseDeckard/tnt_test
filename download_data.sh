#!/bin/bash

DATA_PROCESSED="./data/processed"
DOWNLOAD_URL_PROCESSED_DOCS="https://www.dropbox.com/scl/fi/amfvjamsruknn3q2acuv2/processed_documents.jsonl?rlkey=oq6bvo29twmykwvgn0vnyettr&dl=1"
DOWNLOAD_URL_ID_MAPPING="https://www.dropbox.com/scl/fi/ujjnr64qi6tsvuad4980v/id_mapping.json?rlkey=mlp3vbna1i6vtozv8aq42uqqs&st=thwzj524&dl=1"
DOWNLOAD_URL_EMBEDDINGS="https://www.dropbox.com/scl/fi/9chixpy4h66ch03swad2e/embeddings.npy?rlkey=56j0y3jg9zxgvucrv428bacxt&st=0xw4yrj9&dl=1"
DOWNLOAD_URL_GAZETA="https://www.dropbox.com/scl/fi/yitpf41jk75uina0bfm9d/gazeta_test.jsonl?rlkey=tvkgvl1vkaax0ssh4ma4xzpkq&st=no6zk1m6&dl=1"

mkdir -p "$DATA_PROCESSED"
FILES=(
    "$DATA_PROCESSED/processed_documents.jsonl"
    "$DATA_PROCESSED/id_mapping.json"
    "$DATA_PROCESSED/embeddings.npy"
)
URLS=(
    "$DOWNLOAD_URL_PROCESSED_DOCS"
    "$DOWNLOAD_URL_ID_MAPPING"
    "$DOWNLOAD_URL_EMBEDDINGS"
)

for i in ${!FILES[@]}; do
  if [ ! -f "${FILES[$i]}" ]; then
    echo "Файл ${FILES[$i]} отсутствует. Загружаем..."
    wget -O "${FILES[$i]}" "${URLS[$i]}"
  else
    echo "Файл ${FILES[$i]} уже существует."
  fi
done

DATA_RAW="./data/raw"
mkdir -p "$DATA_RAW"
if [ ! -f "$DATA_RAW/gazeta_test.jsonl" ]; then
  echo "Файл $DATA_RAW/gazeta_test.jsonl отсутствует. Загружаем..."
  wget -O "$DATA_RAW/gazeta_test.jsonl" "$DOWNLOAD_URL_GAZETA"
else
  echo "Файл $DATA_RAW/gazeta_test.jsonl уже существует."
fi
