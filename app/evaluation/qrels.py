import json
from pathlib import Path


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    """Load a query-to-relevant-URL mapping and validate its shape."""
    with Path(path).open(encoding="utf-8") as source:
        raw_qrels = json.load(source)

    if not isinstance(raw_qrels, dict) or not raw_qrels:
        raise ValueError("Qrels must be a non-empty JSON object")

    qrels = {}
    for query, relevant_urls in raw_qrels.items():
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Every qrels key must be a non-empty query")
        if (
            not isinstance(relevant_urls, list)
            or not relevant_urls
            or not all(isinstance(url, str) and url for url in relevant_urls)
        ):
            raise ValueError(f"Qrels for {query!r} must contain relevant URLs")
        qrels[query] = set(relevant_urls)
    return qrels
