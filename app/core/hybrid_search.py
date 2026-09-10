import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol

SearchResult = dict[str, Any]
Weights = Mapping[str, float]


class SearchBackend(Protocol):
    """Minimal interface required by the hybrid searcher."""

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


class HybridSearch:
    """Combine BM25 and vector results without per-request shared state."""

    DEFAULT_WEIGHTS: ClassVar[Weights] = {"bm25": 0.6, "vector": 0.4}

    def __init__(
        self,
        text_search: SearchBackend,
        vector_search: SearchBackend,
        weights: Weights | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.text_search = text_search
        self.vector_search = vector_search
        initial_weights = self.DEFAULT_WEIGHTS if weights is None else weights
        self.weights = self._normalize_weights(initial_weights)

    @staticmethod
    def _normalize_weights(weights: Weights) -> dict[str, float]:
        """Validate weights and return a normalized copy."""
        if set(weights) != {"bm25", "vector"}:
            raise ValueError("Weights must contain exactly 'bm25' and 'vector'")

        values = {name: float(value) for name, value in weights.items()}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Weights must be finite and non-negative")

        total = sum(values.values())
        if total <= 0:
            raise ValueError("At least one weight must be greater than zero")

        return {name: value / total for name, value in values.items()}

    def set_weights(self, weights: Weights) -> None:
        """Set default weights for callers using the legacy stateful API."""
        self.weights = self._normalize_weights(weights)

    def batch_search(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        weights: Weights | None = None,
    ) -> list[list[SearchResult]]:
        """Search multiple queries using one immutable weight snapshot."""
        requested_weights = self.weights if weights is None else weights
        effective_weights = self._normalize_weights(requested_weights)
        return [
            self.search(query, top_k=top_k, weights=effective_weights)
            for query in queries
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        weights: Weights | None = None,
    ) -> list[SearchResult]:
        """Search both backends and combine their scores."""
        requested_weights = self.weights if weights is None else weights
        effective_weights = self._normalize_weights(requested_weights)
        candidate_count = top_k * 2
        text_results = self.text_search.search(query, top_k=candidate_count)
        vector_results = self.vector_search.search(query, top_k=candidate_count)
        return self._combine_results(
            text_results,
            vector_results,
            top_k,
            effective_weights,
        )

    @staticmethod
    def _combine_results(
        bm25_results: Sequence[SearchResult],
        vector_results: Sequence[SearchResult],
        top_k: int,
        weights: Weights,
    ) -> list[SearchResult]:
        """Merge results by URL and return the highest combined scores."""
        doc_scores: dict[str, dict[str, Any]] = {}

        for result in bm25_results:
            url = result["url"]
            doc_scores[url] = {
                "score": result["score"] * weights["bm25"],
                "data": {**result, "score_type": "bm25"},
            }

        for result in vector_results:
            url = result["url"]
            weighted_score = result["score"] * weights["vector"]
            if url in doc_scores:
                doc_scores[url]["score"] += weighted_score
                doc_scores[url]["data"]["score_type"] = "hybrid"
            else:
                doc_scores[url] = {
                    "score": weighted_score,
                    "data": {**result, "score_type": "vector"},
                }

        ranked = sorted(
            doc_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]
        return [{**item["data"], "score": round(item["score"], 4)} for item in ranked]
