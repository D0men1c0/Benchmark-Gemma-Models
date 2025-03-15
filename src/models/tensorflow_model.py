from transformers import TFAutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any, Optional
from logging import getLogger

from models.base_model_loader import BaseModelLoader

logger = getLogger(__name__)

class TensorFlowModelLoader(BaseModelLoader):
    """
    Class for loading models using TensorFlow with optional quantization.
    """
    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the TensorFlow model loader.

        :param model_name: Name of the model to load.
        :param kwargs: Additional arguments for the model loader.
        """
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a TensorFlow model and tokenizer with optional quantization.

        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        if quantization:
            logger.warning(f"Quantization ({quantization}) is not yet supported for TensorFlow models.")
        
        model = TFAutoModelForCausalLM.from_pretrained(self.model_name, **self.kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer