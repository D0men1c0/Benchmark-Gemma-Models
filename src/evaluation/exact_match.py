from typing import Any

from evaluation.base_model_evaluator import BaseMetric


class ExactMatchMetric(BaseMetric):
    """
    Class for computing exact match.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute exact match.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options.
        :return: Exact match score.
        """
        return float(predictions == labels)