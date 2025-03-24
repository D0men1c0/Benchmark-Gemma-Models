from typing import Any, Optional, Type, Dict

from .base_model_loader import BaseModelLoader
from .concrete_models import HuggingFaceModelLoader, PyTorchModelLoader, TensorFlowModelLoader
from utils.logger import setup_logger

logger = setup_logger()

class ModelLoaderFactory:
    """
    Factory class to create model loaders using a registry pattern.
    """
    _LOADER_REGISTRY: Dict[str, Type[BaseModelLoader]] = {
        "huggingface": HuggingFaceModelLoader,
        "pytorch": PyTorchModelLoader,
        "tensorflow": TensorFlowModelLoader
    }

    @classmethod
    def register_loader(cls, framework: str, loader_class: Type[BaseModelLoader]):
        """
        Register a new model loader type for a framework.
        
        :param framework: Framework identifier string
        :param loader_class: Model loader class to register
        """
        cls._LOADER_REGISTRY[framework.lower()] = loader_class

    @classmethod
    def get_model_loader(
        cls,
        model_name: str,
        framework: str,
        quantization: Optional[str] = None,
        **kwargs: Any
    ) -> BaseModelLoader:
        """
        Get initialized model loader with case-insensitive lookup.
        
        :param model_name: Name/path of the model
        :param framework: Target framework (case-insensitive)
        :param quantization: Optional quantization method
        :param kwargs: Additional loader-specific parameters
        :return: Initialized model loader instance
        """
        framework = framework.lower()
        loader_class = cls._LOADER_REGISTRY.get(framework)
        logger.info(f"Loading model using {framework} framework")
        if not loader_class:
            available = ", ".join(cls._LOADER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported framework: {framework}. "
                f"Available options: {available}"
            )
        
        return loader_class(
            model_name=model_name,
            quantization=quantization,
            **kwargs
        )