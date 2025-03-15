from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any, Optional

from models.base_model_loader import BaseModelLoader

class HuggingFaceModelLoader(BaseModelLoader):
    """
    Class for loading models from Hugging Face with optional quantization.
    """
    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the Hugging Face model loader.

        :param model_name: Name of the model to load.
        :param kwargs: Additional arguments for the model loader.
        """
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a Hugging Face model and tokenizer with optional quantization.

        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.kwargs["quantization_config"] = quantization_config
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer