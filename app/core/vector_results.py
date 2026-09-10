from collections.abc import Sequence
from typing import Any


def validate_artifacts(document_count: int, embedding_shape: tuple[int, ...]) -> None:
    """Reject inconsistent search artifacts before constructing an index."""
    if len(embedding_shape) != 2:
        raise ValueError("Embeddings must be a two-dimensional array")
    if embedding_shape[0] != document_count:
        raise ValueError(
            "Document and embedding counts differ: "
            f"{document_count} != {embedding_shape[0]}"
        )
    if document_count == 0 or embedding_shape[1] == 0:
        raise ValueError("Search artifacts must not be empty")


def bounded_top_k(top_k: int, document_count: int) -> int:
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    return min(top_k, document_count)


def format_results(
    documents: Sequence[dict[str, Any]],
    indices: Sequence[int],
    scores: Sequence[float],
) -> list[dict[str, Any]]:
    """Map valid FAISS hits to API results without negative-index leakage."""
    results = []
    for index, score in zip(indices, scores):
        if index < 0 or index >= len(documents):
            continue
        document = documents[index]
        results.append(
            {
                "title": document["title"],
                "summary": document["summary"],
                "url": document["url"],
                "date": document["date"],
                "score": float(score),
            }
        )
    return results
