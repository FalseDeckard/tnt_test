import math
from collections.abc import Iterable, Sequence


def _validate(relevant: set[str], k: int) -> None:
    if not relevant:
        raise ValueError("At least one relevant document is required")
    if k < 1:
        raise ValueError("k must be greater than zero")


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("At least one relevant document is required")
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant_set:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    relevant_set = set(relevant)
    _validate(relevant_set, k)
    return len(set(retrieved[:k]) & relevant_set) / len(relevant_set)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Calculate binary normalized discounted cumulative gain."""
    relevant_set = set(relevant)
    _validate(relevant_set, k)
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
        if document_id in relevant_set
    )
    ideal_count = min(len(relevant_set), k)
    ideal_gain = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return gain / ideal_gain


def evaluate_ranking(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> dict[str, float]:
    """Calculate standard binary relevance metrics for one query."""
    relevant_set = set(relevant)
    return {
        "mrr": reciprocal_rank(retrieved[:k], relevant_set),
        "recall": recall_at_k(retrieved, relevant_set, k),
        "ndcg": ndcg_at_k(retrieved, relevant_set, k),
    }
