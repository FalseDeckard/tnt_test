import asyncio
import unittest

from fastapi import HTTPException

from app.api.endpoints import SearchManager, readiness
from app.models.schemas import SearchQuery


class FakeSearcher:
    def search(self, query, top_k):
        return []


class SearchManagerTests(unittest.TestCase):
    def test_uninitialized_manager_is_not_ready(self):
        with self.assertRaises(HTTPException) as context:
            readiness(SearchManager())
        self.assertEqual(context.exception.status_code, 503)

    def test_search_returns_response_from_injected_backend(self):
        manager = SearchManager()
        manager.searchers = {"text": FakeSearcher()}
        query = SearchQuery(queries=["test"], method="text")

        response = asyncio.run(manager.search(query))

        self.assertEqual(response.total_queries, 1)
        self.assertEqual(response.method, "text")
        self.assertEqual(response.results[0].query, "test")


if __name__ == "__main__":
    unittest.main()
