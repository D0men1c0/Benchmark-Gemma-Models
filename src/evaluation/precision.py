from typing import Any
from sklearn.metrics import precision_score

from evaluation.base_model_evaluator import BaseMetric

class PrecisionMetric(BaseMetric):
    """
    Class for computing precision.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute precision.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: Precision score.
        """
        average = kwargs.get("average", "weighted")
        return precision_score(labels, predictions, average=average)