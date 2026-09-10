import json
import tempfile
import unittest
from pathlib import Path

from app.data.documents import read_documents, write_documents


class DocumentArtifactTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_documents(self) -> None:
        documents = [
            {"url": "one", "title": "Экономика"},
            {"url": "two", "title": "Технологии"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "documents.jsonl"
            write_documents(path, documents)

            self.assertEqual(read_documents(path), documents)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_reader_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text('\n{"url": "one"}\n\n', encoding="utf-8")

            self.assertEqual(read_documents(path), [{"url": "one"}])

    def test_writer_atomically_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            write_documents(path, [{"url": "old"}])

            write_documents(path, [{"url": "new"}])

            self.assertEqual(read_documents(path), [{"url": "new"}])

    def test_reader_reports_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text('{}\n{"broken"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                read_documents(path)

    def test_reader_rejects_non_object_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "Expected an object"):
                read_documents(path)


if __name__ == "__main__":
    unittest.main()
