from typing import Dict, List, Optional
from dataclasses import dataclass
from lm_evaluation_harness.tasks import Task

@dataclass
class MMLUBenchmark:
    """MMLU (Massive Multitask Language Understanding) benchmark."""
    subjects: Optional[List[str]] = None  # None = all subjects

    def get_task_config(self) -> Dict:
        """Return the task configuration for lm-evaluation-harness."""
        if self.subjects:
            return {
                "task": f"mmlu_{'_'.join(self.subjects)}",
                "num_fewshot": 5,
                "split": "test"
            }
        return {
            "task": "mmlu",
            "num_fewshot": 5,
            "split": "test"
        }
