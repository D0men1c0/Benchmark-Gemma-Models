from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

class BaseModelLoader(ABC):
    """
    Abstract base class for loading models from different frameworks.
    """
    @abstractmethod
    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load the model and tokenizer.

        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        pass