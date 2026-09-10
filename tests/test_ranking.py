import unittest

from app.core.ranking import normalize_positive_scores


class RankingTests(unittest.TestCase):
    def test_normalizes_selected_positive_scores(self):
        self.assertEqual(
            normalize_positive_scores([0.5, 2.0, 1.0], [1, 2]),
            [(1, 1.0), (2, 0.5)],
        )

    def test_omits_zero_and_negative_scores(self):
        self.assertEqual(normalize_positive_scores([0.0, -1.0], [0, 1]), [])


if __name__ == "__main__":
    unittest.main()
