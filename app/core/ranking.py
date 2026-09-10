from collections.abc import Sequence


def normalize_positive_scores(
    scores: Sequence[float],
    indices: Sequence[int],
) -> list[tuple[int, float]]:
    """Normalize positive retrieval scores against the best selected score."""
    selected = [(int(index), float(scores[index])) for index in indices]
    selected = [(index, score) for index, score in selected if score > 0]
    if not selected:
        return []

    maximum = max(score for _, score in selected)
    return [(index, score / maximum) for index, score in selected]
