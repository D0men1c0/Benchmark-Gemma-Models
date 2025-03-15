from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any, Optional

from models.base_model_loader import BaseModelLoader
from logging import getLogger

logger = getLogger(__name__)

class PyTorchModelLoader(BaseModelLoader):
    """
    Class for loading models using PyTorch with optional quantization.
    """
    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the PyTorch model loader.

        :param model_name: Name of the model to load.
        :param kwargs: Additional arguments for the model loader.
        """
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a PyTorch model and tokenizer with optional quantization.

        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        if quantization == "4bit" or quantization == "8bit":
            logger.warning(f"Quantization ({quantization}) is not yet supported for PyTorch models in this implementation.")
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer