import unittest

from app.evaluation.metrics import (
    evaluate_ranking,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_reciprocal_rank_uses_first_relevant_result(self):
        self.assertEqual(reciprocal_rank(["a", "b", "c"], {"b", "c"}), 0.5)
        self.assertEqual(reciprocal_rank(["a"], {"b"}), 0.0)

    def test_recall_at_k_counts_unique_relevant_documents(self):
        self.assertEqual(recall_at_k(["a", "a", "b"], {"a", "b"}, 2), 0.5)
        self.assertEqual(recall_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_ndcg_rewards_better_ordering(self):
        ideal = ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3)
        delayed = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
        self.assertEqual(ideal, 1.0)
        self.assertLess(delayed, ideal)

    def test_evaluate_ranking_respects_cutoff(self):
        metrics = evaluate_ranking(["x", "a"], {"a"}, 1)
        self.assertEqual(metrics, {"mrr": 0.0, "recall": 0.0, "ndcg": 0.0})

    def test_metrics_reject_missing_labels(self):
        with self.assertRaisesRegex(ValueError, "relevant document"):
            evaluate_ranking(["a"], set(), 1)


if __name__ == "__main__":
    unittest.main()
