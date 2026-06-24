from typing import Dict, List
import numpy as np

class AccuracyMetric:
    """Compute accuracy from model predictions."""

    @staticmethod
    def compute(predictions: List[str], references: List[str]) -> float:
        """Compute accuracy between predictions and references."""
        correct = 0
        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
        return correct / len(predictions) if predictions else 0.0
