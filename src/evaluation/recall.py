from typing import Any
from sklearn.metrics import recall_score

from evaluation.base_model_evaluator import BaseMetric

class RecallMetric(BaseMetric):
    """
    Class for computing recall.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute recall.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: Recall score.
        """
        average = kwargs.get("average", "weighted")
        return recall_score(labels, predictions, average=average)