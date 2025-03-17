from typing import Any
from sklearn.metrics import f1_score

from evaluation.base_model_evaluator import BaseMetric

class F1ScoreMetric(BaseMetric):
    """
    Class for computing F1 score.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute F1 score.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: F1 score.
        """
        average = kwargs.get("average", "weighted")
        return f1_score(labels, predictions, average=average)