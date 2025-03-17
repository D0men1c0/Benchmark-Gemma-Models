from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    @abstractmethod
    def compute(self, predictions: Any, labels: Any) -> float:
        """
        Compute the metric value.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :return: Computed metric value.
        """
        pass