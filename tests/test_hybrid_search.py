import unittest

from app.core.hybrid_search import HybridSearch


def result(url: str, score: float) -> dict:
    return {
        "title": url,
        "summary": "summary",
        "url": url,
        "date": "2026-01-01",
        "score": score,
    }


class FakeSearch:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class HybridSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = FakeSearch([result("text", 1.0), result("shared", 0.5)])
        self.vector = FakeSearch([result("vector", 1.0), result("shared", 0.5)])
        self.search = HybridSearch(self.text, self.vector)

    def test_request_weights_do_not_mutate_defaults(self) -> None:
        defaults = self.search.weights.copy()

        text_first = self.search.search(
            "query",
            top_k=1,
            weights={"bm25": 0.9, "vector": 0.1},
        )
        vector_first = self.search.search(
            "query",
            top_k=1,
            weights={"bm25": 0.1, "vector": 0.9},
        )

        self.assertEqual(text_first[0]["url"], "text")
        self.assertEqual(vector_first[0]["url"], "vector")
        self.assertEqual(self.search.weights, defaults)

    def test_combines_shared_documents(self) -> None:
        results = self.search.search(
            "query",
            top_k=3,
            weights={"bm25": 0.5, "vector": 0.5},
        )

        shared = next(item for item in results if item["url"] == "shared")
        self.assertEqual(shared["score"], 0.5)
        self.assertEqual(shared["score_type"], "hybrid")
        self.assertEqual(self.text.calls, [("query", 6)])
        self.assertEqual(self.vector.calls, [("query", 6)])

    def test_normalizes_weights(self) -> None:
        weights = HybridSearch._normalize_weights({"bm25": 3, "vector": 1})
        self.assertEqual(weights, {"bm25": 0.75, "vector": 0.25})

    def test_rejects_zero_total_weight(self) -> None:
        with self.assertRaises(ValueError):
            HybridSearch._normalize_weights({"bm25": 0, "vector": 0})

    def test_rejects_empty_weights(self) -> None:
        with self.assertRaises(ValueError):
            HybridSearch(self.text, self.vector, weights={})

    def test_backend_failures_are_not_hidden(self) -> None:
        class FailingSearch:
            def search(self, query, top_k):
                raise RuntimeError("backend failed")

        search = HybridSearch(FailingSearch(), FakeSearch([]))
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            search.search("query")


if __name__ == "__main__":
    unittest.main()
