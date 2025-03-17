import numpy as np
from typing import Any

from evaluation.base_model_evaluator import BaseMetric

class PerplexityMetric(BaseMetric):
    """
    Class for computing perplexity.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute perplexity.

        :param predictions: Model predictions (log probabilities).
        :param labels: Ground truth labels.
        :param kwargs: Additional options.
        :return: Perplexity score.
        """
        log_probs = np.log(predictions)
        perplexity = np.exp(-np.mean(log_probs))
        return perplexity