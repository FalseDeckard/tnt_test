#!/usr/bin/env bash
set -Eeuo pipefail

DATA_PROCESSED="./data/processed"
DATA_RAW="./data/raw"
mkdir -p "$DATA_PROCESSED" "$DATA_RAW"

FILES=(
  "$DATA_PROCESSED/processed_documents.jsonl"
  "$DATA_PROCESSED/id_mapping.json"
  "$DATA_PROCESSED/embeddings.npy"
  "$DATA_RAW/gazeta_test.jsonl"
)
URLS=(
  "https://www.dropbox.com/scl/fi/amfvjamsruknn3q2acuv2/processed_documents.jsonl?rlkey=oq6bvo29twmykwvgn0vnyettr&dl=1"
  "https://www.dropbox.com/scl/fi/ujjnr64qi6tsvuad4980v/id_mapping.json?rlkey=mlp3vbna1i6vtozv8aq42uqqs&st=thwzj524&dl=1"
  "https://www.dropbox.com/scl/fi/9chixpy4h66ch03swad2e/embeddings.npy?rlkey=56j0y3jg9zxgvucrv428bacxt&st=0xw4yrj9&dl=1"
  "https://www.dropbox.com/scl/fi/yitpf41jk75uina0bfm9d/gazeta_test.jsonl?rlkey=tvkgvl1vkaax0ssh4ma4xzpkq&st=no6zk1m6&dl=1"
)
CHECKSUMS=(
  "8af29e75999b410e71823f2cf70387b8cbaa4a9b73319aa2c53774b714adbe4f"
  "90638cf4474ec3f99da2ac27687163ef456d312912e14fcbcfb999f2d8534b0d"
  "0e2b92b7b7ef88b25b78658d839aefbeb794a7c0923cc6ba4f4d3edf60f08d8b"
  "3963ca7e2313c4bb75a4140abd614e17d98199c9f03f03490ab6afb19bfbf6cf"
)

verify_checksum() {
  local file="$1"
  local expected="$2"
  local actual

  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
  else
    echo "Neither sha256sum nor shasum is available" >&2
    return 1
  fi

  [[ "$actual" == "$expected" ]]
}

download_verified() {
  local target="$1"
  local url="$2"
  local checksum="$3"
  local partial="${target}.part"

  if [[ -f "$target" ]] && verify_checksum "$target" "$checksum"; then
    echo "Verified existing file: $target"
    return
  fi

  rm -f "$partial"
  if ! wget --https-only --tries=3 --timeout=30 --retry-connrefused \
    --output-document="$partial" "$url"; then
    rm -f "$partial"
    return 1
  fi

  if ! verify_checksum "$partial" "$checksum"; then
    echo "Checksum mismatch: $target" >&2
    rm -f "$partial"
    return 1
  fi
  mv "$partial" "$target"
  echo "Downloaded and verified: $target"
}

for index in "${!FILES[@]}"; do
  download_verified "${FILES[$index]}" "${URLS[$index]}" "${CHECKSUMS[$index]}"
done
