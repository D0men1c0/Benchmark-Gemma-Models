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

        :param predictions: Model predictions (probabilities, not text).
        :param labels: Ground truth labels.
        :param kwargs: Additional options.
        :return: Perplexity score.
        """
        predictions = np.array(predictions, dtype=np.float32)

        # Ensure probabilities are in (0, 1] to avoid log(0) errors
        predictions = np.clip(predictions, 1e-10, 1.0)

        log_probs = np.log(predictions)
        perplexity = np.exp(-np.mean(log_probs))

        return perplexity