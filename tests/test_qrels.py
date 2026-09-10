import json
import tempfile
import unittest
from pathlib import Path

from app.evaluation.qrels import load_qrels


class QrelsTests(unittest.TestCase):
    def write_qrels(self, value):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "qrels.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_load_qrels_deduplicates_relevant_urls(self):
        path = self.write_qrels({"query": ["url-1", "url-1", "url-2"]})
        self.assertEqual(load_qrels(path), {"query": {"url-1", "url-2"}})

    def test_load_qrels_rejects_empty_mapping(self):
        with self.assertRaisesRegex(ValueError, "non-empty JSON object"):
            load_qrels(self.write_qrels({}))

    def test_load_qrels_rejects_queries_without_labels(self):
        with self.assertRaisesRegex(ValueError, "relevant URLs"):
            load_qrels(self.write_qrels({"query": []}))


if __name__ == "__main__":
    unittest.main()
