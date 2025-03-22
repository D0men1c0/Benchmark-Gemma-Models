from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

class TaskHandler(ABC):
    """Abstract base class for task handlers."""

    def __init__(self, model: Any, tokenizer: Any, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def process_example(self, example: Dict[str, Any]) -> Tuple[Any, Optional[Any]]:
        """
        Process a single example and return the prediction and label.

        :param example: A dictionary representing a single dataset example.
        :return: Tuple containing the prediction and label (label may be None).
        """
        pass