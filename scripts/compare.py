import argparse
from evaluation.evaluator import LLEvaluator

def main():
    parser = argparse.ArgumentParser(description="Compare multiple models on multiple benchmarks.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names (from config/models.json)")
    parser.add_argument("--benchmarks", nargs="+", required=True, help="Benchmark names (from config/benchmarks.json)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for testing)")
    parser.add_argument("--report-format", type=str, default="markdown", choices=["markdown", "csv", "json"])
    parser.add_argument("--report-name", type=str, default="comparison_report")
    args = parser.parse_args()

    evaluator = LLEvaluator()
    results = evaluator.evaluate_models(
        model_names=args.models,
        benchmark_names=args.benchmarks,
        limit=args.limit
    )

    report = evaluator.generate_report(results, output_format=args.report_format)
    evaluator.save_report(report, args.report_name, args.report_format)

    print("\n Comparison Report:")
    print(report)

if __name__ == "__main__":
    main()
