import argparse
from evaluation.evaluator import LLEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate an LLM on a benchmark.")
    parser.add_argument("--model", type=str, required=True, help="Model name (from config/models.json)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (from config/benchmarks.json)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for testing)")
    parser.add_argument("--save-raw", action="store_true", help="Save raw results")
    args = parser.parse_args()

    evaluator = LLEvaluator()
    result = evaluator.evaluate_model(
        model_name=args.model,
        benchmark_name=args.benchmark,
        limit=args.limit,
        save_raw=args.save_raw
    )

    print("\n Results:")
    for metric, value in result.metrics.items():
        print(f"- **{metric}**: {value:.4f}")

if __name__ == "__main__":
    main()
