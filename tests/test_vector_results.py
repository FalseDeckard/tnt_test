import unittest

from app.core.vector_results import bounded_top_k, format_results, validate_artifacts


class VectorResultTests(unittest.TestCase):
    def test_validate_artifacts_accepts_matching_matrix(self):
        validate_artifacts(2, (2, 4))

    def test_validate_artifacts_rejects_mismatched_counts(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            validate_artifacts(2, (3, 4))

    def test_bounded_top_k_clamps_to_document_count(self):
        self.assertEqual(bounded_top_k(10, 3), 3)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            bounded_top_k(0, 3)

    def test_format_results_skips_invalid_faiss_indices(self):
        documents = [
            {"title": "A", "summary": "B", "url": "u", "date": "d"},
        ]

        results = format_results(documents, [-1, 0, 8], [0.9, 0.8, 0.7])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "u")
        self.assertEqual(results[0]["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
