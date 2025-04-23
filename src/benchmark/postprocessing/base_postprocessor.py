from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

class BasePostProcessor(ABC):
    """Abstract base class for post-processing model outputs."""

    @abstractmethod
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[str], List[str]]:
        """
        Processes raw model predictions and labels into a format suitable
        for specific evaluation metrics based on the task.

        :param predictions: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional, might be needed for context like choices in MMLU).
        :return: Tuple of (processed_predictions, processed_labels).
                 Both lists should contain strings ready for metric comparison.
        """
        pass