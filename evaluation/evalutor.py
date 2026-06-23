import json
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from lm_evaluation_harness import evaluate as lm_evaluate
from lm_evaluation_harness.evaluators import simple_evaluate

@dataclass
class BenchmarkResult:
    model: str
    benchmark: str
    metrics: Dict[str, float]
    raw_results: Optional[Dict] = None

class LLEvaluator:
    def __init__(
        self,
        benchmarks_config: str = "config/benchmarks.json",
        models_config: str = "config/models.json",
        results_dir: str = "results"
    ):
        self.benchmarks_config = self._load_json(benchmarks_config)
        self.models_config = self._load_json(models_config)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create raw and reports directories
        (self.results_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "reports").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_json(path: str) -> Dict:
        with open(path) as f:
            return json.load(f)

    def _get_model_and_tokenizer(self, model_name: str):
        """Load model and tokenizer based on config."""
        model_config = self.models_config[model_name]
        model_type = model_config["type"]

        if model_type == "huggingface":
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_config["name"])

            # Load model
            if model_config.get("quantization") == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["name"],
                    quantization_config=bnb_config,
                    device_map=model_config.get("device_map", "auto"),
                    torch_dtype=torch.float16,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["name"],
                    device_map=model_config.get("device_map", "auto"),
                    torch_dtype=torch.float16,
                )

            # Load PeftModel if it's a fine-tuned model
            if "/final" in model_config["name"] or model_config.get("peft"):
                model = PeftModel.from_pretrained(model, model_config["name"])

            return model, tokenizer

        elif model_type == "openai":
            # For OpenAI API models, return None for model/tokenizer
            # (handled by lm-evaluation-harness)
            return None, None

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _get_benchmark_tasks(self, benchmark_name: str) -> List[str]:
        """Get the lm-evaluation-harness task names for a benchmark."""
        benchmark_config = self.benchmarks_config[benchmark_name]
        if "subjects" in benchmark_config and benchmark_config["subjects"]:
            # For MMLU with specific subjects
            return [
                f"{benchmark_config['task']}_{subject}"
                for subject in benchmark_config["subjects"]
            ]
        return [benchmark_config["task"]]

    def evaluate_model(
        self,
        model_name: str,
        benchmark_name: str,
        limit: Optional[int] = None,
        save_raw: bool = True
    ) -> BenchmarkResult:
        """Evaluate a single model on a single benchmark."""
        print(f"🔍 Evaluating **{model_name}** on **{benchmark_name}**...")

        # Get benchmark config
        benchmark_config = self.benchmarks_config[benchmark_name]
        tasks = self._get_benchmark_tasks(benchmark_name)

        # Prepare evaluation args
        eval_args = {
            "model": model_name if self.models_config[model_name]["type"] == "openai" else None,
            "model_args": f"pretrained={self.models_config[model_name]['name']}" if self.models_config[model_name]["type"] == "huggingface" else None,
            "tasks": tasks,
            "num_fewshot": benchmark_config["num_fewshot"],
            "batch_size": "auto",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "limit": limit,
            "write_out": False,
        }

        # Handle OpenAI API key
        if self.models_config[model_name]["type"] == "openai":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", self.models_config[model_name].get("api_key"))
            eval_args["model"] = model_name

        # Run evaluation
        results = lm_evaluate(**eval_args)

        # Extract metrics
        metrics = {}
        for task in tasks:
            if task in results["results"]:
                task_results = results["results"][task]
                for metric in benchmark_config["metrics"]:
                    if metric in task_results:
                        # Handle nested metrics (e.g., "accuracy" vs "acc,none")
                        if isinstance(task_results[metric], dict):
                            # Take the default metric (e.g., "none" for acc,none)
                            metric_key = next(iter(task_results[metric]))
                            metrics[metric] = task_results[metric][metric_key]
                        else:
                            metrics[metric] = task_results[metric]

        # Save raw results
        if save_raw:
            raw_path = self.results_dir / "raw" / f"{model_name}_{benchmark_name}.json"
            with open(raw_path, "w") as f:
                json.dump(results, f, indent=2)

        return BenchmarkResult(
            model=model_name,
            benchmark=benchmark_name,
            metrics=metrics,
            raw_results=results
        )

    def evaluate_models(
        self,
        model_names: List[str],
        benchmark_names: List[str],
        limit: Optional[int] = None
    ) -> List[BenchmarkResult]:
        """Evaluate multiple models on multiple benchmarks."""
        results = []
        for model_name in model_names:
            for benchmark_name in benchmark_names:
                try:
                    result = self.evaluate_model(model_name, benchmark_name, limit, save_raw=True)
                    results.append(result)
                    print(f"✅ Completed: {model_name} on {benchmark_name}")
                except Exception as e:
                    print(f"❌ Failed: {model_name} on {benchmark_name} - {str(e)}")
        return results

    def generate_report(
        self,
        results: List[BenchmarkResult],
        output_format: str = "markdown",
        filename: Optional[str] = None
    ) -> str:
        """Generate a report from evaluation results."""
        # Convert to DataFrame
        data = []
        for result in results:
            for metric, value in result.metrics.items():
                data.append({
                    "Model": result.model,
                    "Benchmark": result.benchmark,
                    "Metric": metric,
                    "Value": value
                })

        df = pd.DataFrame(data)

        if output_format == "markdown":
            # Pivot for better display
            pivot_df = df.pivot_table(
                index=["Model", "Benchmark"],
                columns="Metric",
                values="Value"
            ).reset_index()

            # Round values
            for col in pivot_df.columns:
                if col not in ["Model", "Benchmark"]:
                    pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

            # Generate markdown table
            md_table = pivot_df.to_markdown(index=False)
            report = f"# 📊 LLM Evaluation Report\n\n{md_table}\n\n---\n*Generated by [llm-evaluation-suite](https://github.com/HaaseSchuetz/llm-evaluation-suite)*"
            return report

        elif output_format == "csv":
            return df.to_csv(index=False)

        elif output_format == "json":
            return df.to_json(orient="records", indent=2)

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def save_report(self, report: str, filename: str, output_format: str = "markdown"):
        """Save a report to file."""
        ext = {
            "markdown": "md",
            "csv": "csv",
            "json": "json"
        }[output_format]
        filepath = self.results_dir / "reports" / f"{filename}.{ext}"
        with open(filepath, "w") as f:
            f.write(report)
        print(f"💾 Report saved to: {filepath}")
        return filepath
