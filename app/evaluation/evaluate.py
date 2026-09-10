import argparse
import asyncio
import inspect
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.endpoints import SearchManager
from app.data.documents import read_documents
from app.evaluation.metrics import evaluate_ranking
from app.evaluation.qrels import load_qrels
from app.models.schemas import SearchQuery

logger = logging.getLogger(__name__)


class DatasetEvaluator:
    """Evaluate search methods against explicit binary relevance judgments."""

    def __init__(
        self,
        search_manager: SearchManager,
        qrels_path: str | Path,
        data_dir: str | Path = "data/processed",
        methods: tuple[str, ...] = ("vector", "text", "hybrid"),
        k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> None:
        self.search_manager = search_manager
        self.qrels = load_qrels(qrels_path)
        self.documents = read_documents(Path(data_dir) / "processed_documents.jsonl")
        self.methods = methods
        self.k_values = k_values

    async def evaluate_method(self, method: str) -> dict[str, Any]:
        measurements = {k: [] for k in self.k_values}
        timings = []

        for query, relevant_urls in self.qrels.items():
            response = await self.search_manager.search(
                SearchQuery(
                    queries=[query],
                    method=method,
                    top_k=max(self.k_values),
                )
            )
            result = response.results[0]
            retrieved_urls = [document.url for document in result.results]
            timings.append(result.time_taken)
            for k in self.k_values:
                measurements[k].append(
                    evaluate_ranking(retrieved_urls, relevant_urls, k)
                )

        metrics = {}
        for k, query_metrics in measurements.items():
            metrics[str(k)] = {
                name: statistics.fmean(row[name] for row in query_metrics)
                for name in ("mrr", "recall", "ndcg")
            }

        return {
            "latency_seconds": {
                "mean": statistics.fmean(timings),
                "median": statistics.median(timings),
            },
            "metrics": metrics,
        }

    async def run(self) -> dict[str, Any]:
        results = {}
        for method in self.methods:
            logger.info("Evaluating %s search", method)
            results[method] = await self.evaluate_method(method)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_size": len(self.documents),
            "query_count": len(self.qrels),
            "results": results,
        }


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = directory / f"search_evaluation_{timestamp}.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary.replace(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate search ranking quality")
    parser.add_argument("--qrels", required=True, help="Path to qrels JSON")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/evaluation")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    manager = SearchManager(data_dir=args.data_dir)
    initialization = manager.initialize()
    if inspect.isawaitable(initialization):
        await initialization
    try:
        evaluator = DatasetEvaluator(manager, args.qrels, args.data_dir)
        report_path = write_report(await evaluator.run(), args.output_dir)
        print(f"Report saved to {report_path}")
    finally:
        manager.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
