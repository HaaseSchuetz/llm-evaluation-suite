import argparse
import json
from pathlib import Path
from evaluation.evaluator import LLEvaluator, BenchmarkResult

def load_raw_results(results_dir: str = "results/raw") -> List[BenchmarkResult]:
    """Load all raw results from the results directory."""
    results = []
    raw_dir = Path(results_dir)
    for file in raw_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        # Extract model and benchmark from filename (e.g., "mistral-7b_mmlu.json")
        model, benchmark = file.stem.split("_")
        metrics = {}
        if "results" in data:
            for task, task_results in data["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, dict):
                        # Handle nested metrics
                        metric_key = next(iter(value))
                        metrics[metric] = value[metric_key]
                    elif isinstance(value, (int, float)):
                        metrics[metric] = value

        results.append(BenchmarkResult(
            model=model,
            benchmark=benchmark,
            metrics=metrics,
            raw_results=data
        ))
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate a report from raw evaluation results.")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "csv", "json"])
    parser.add_argument("--name", type=str, default="report")
    args = parser.parse_args()

    evaluator = LLEvaluator()
    results = load_raw_results()
    report = evaluator.generate_report(results, output_format=args.format)
    evaluator.save_report(report, args.name, args.format)

    print("\n Report:")
    print(report)

if __name__ == "__main__":
    main()
