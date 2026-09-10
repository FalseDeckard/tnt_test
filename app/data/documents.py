"""Utilities for the newline-delimited JSON document artifact."""

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Document = dict[str, Any]


def read_documents(path: str | Path) -> list[Document]:
    """Read documents from a UTF-8 JSONL file."""
    source = Path(path)
    documents: list[Document] = []

    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source} at line {line_number}"
                ) from exc
            if not isinstance(document, dict):
                raise TypeError(f"Expected an object in {source} at line {line_number}")
            documents.append(document)

    return documents


def write_documents(path: str | Path, documents: Iterable[Document]) -> None:
    """Atomically write documents as UTF-8 JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for document in documents:
                stream.write(json.dumps(document, ensure_ascii=False))
                stream.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
