from typing import Any, Optional

from models.base_model_loader import BaseModelLoader
from models.huggingface_model import HuggingFaceModelLoader
from models.pytorch_model import PyTorchModelLoader
from models.tensorflow_model import TensorFlowModelLoader

class ModelLoaderFactory:
    """
    Factory class to create the appropriate model loader based on the framework.
    """
    @staticmethod
    def get_model_loader(model_name: str, framework: str, quantization: Optional[str] = None, **kwargs: Any) -> BaseModelLoader:
        """
        Get the model loader for the specified framework.

        :param model_name: Name of the model to load.
        :param framework: Framework to use (e.g., "huggingface", "tensorflow", "pytorch").
        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :param kwargs: Additional arguments for the model loader.
        :return: An instance of the appropriate model loader.
        :raises ValueError: If the framework is not supported.
        """
        if framework == "huggingface":
            return HuggingFaceModelLoader(model_name, **kwargs)
        elif framework == "tensorflow":
            return TensorFlowModelLoader(model_name, **kwargs)
        elif framework == "pytorch":
            return PyTorchModelLoader(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")