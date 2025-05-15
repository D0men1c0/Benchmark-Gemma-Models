import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TFAutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple, Any, Optional
from .base_model_loader import BaseModelLoader
from utils.logger import setup_logger

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
        self.logger = setup_logger(self.__class__.__name__)
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        self.logger.info(f"Loading model: {self.model_name} with quantization: {quantization}")
        load_kwargs = self.kwargs.copy()

        torch_dtype_str = load_kwargs.pop("torch_dtype", None)
        actual_torch_dtype = torch.float32  # Default fallback

        if torch_dtype_str == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                actual_torch_dtype = torch.bfloat16
            else:
                self.logger.warning("Configured torch_dtype 'bfloat16' not supported by CUDA, falling back to float32.")
        elif torch_dtype_str == "float16":
            actual_torch_dtype = torch.float16
        elif torch_dtype_str == "float32":
            actual_torch_dtype = torch.float32
        elif torch_dtype_str:
            self.logger.warning(f"Unsupported torch_dtype '{torch_dtype_str}' specified. Defaulting to float32.")
        elif torch_dtype_str is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                actual_torch_dtype = torch.bfloat16

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=actual_torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            load_kwargs["quantization_config"] = quantization_config
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=actual_torch_dtype,
                bnb_8bit_quant_type="dynamic",
                bnb_8bit_use_double_quant=True
            )
            load_kwargs["quantization_config"] = quantization_config
        elif quantization is None:
            self.logger.info(f"Loading model {self.model_name} without quantization. Ensuring device_map='auto' if not already set.")
            if "device_map" not in load_kwargs:
                load_kwargs["device_map"] = "auto"

        if "quantization" in load_kwargs:
            load_kwargs.pop("quantization")
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
    
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
        self.logger = setup_logger(self.__class__.__name__)
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a PyTorch model and tokenizer with optional quantization.

        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        self.logger.info(f"Loading model: {self.model_name} with quantization: {quantization}")
        if quantization == "4bit" or quantization == "8bit":
            self.logger.warning(f"Quantization ({quantization}) is not yet supported for PyTorch models in this implementation.")
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

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
        self.logger = setup_logger(self.__class__.__name__)
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a TensorFlow model and tokenizer with optional quantization.

        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        self.logger.info(f"Loading model: {self.model_name} with quantization: {quantization}")
        if quantization:
            self.logger.warning(f"Quantization ({quantization}) is not yet supported for TensorFlow models.")
        
        model = TFAutoModelForCausalLM.from_pretrained(self.model_name, **self.kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer