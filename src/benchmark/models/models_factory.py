from typing import Any, Optional, Type, Dict

from .base_model_loader import BaseModelLoader
from .concrete_models import HuggingFaceModelLoader, PyTorchModelLoader, TensorFlowModelLoader, CustomScriptModelLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelLoaderFactory:
    """
    Factory class to create model loaders using a registry pattern.
    """
    _LOADER_REGISTRY: Dict[str, Type[BaseModelLoader]] = {
        "huggingface": HuggingFaceModelLoader,
        "pytorch": PyTorchModelLoader,
        "tensorflow": TensorFlowModelLoader,
        "custom_script": CustomScriptModelLoader,
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
        model_specific_config_params: Optional[Dict[str, Any]] = None,
        global_model_creation_params: Optional[Dict[str, Any]] = None
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
        
        effective_kwargs: Dict[str, Any] = {}
        if global_model_creation_params:
            effective_kwargs.update(global_model_creation_params)
        if model_specific_config_params:
            effective_kwargs.update(model_specific_config_params)

        effective_kwargs.pop('name', None)
        effective_kwargs.pop('framework', None)
        effective_kwargs.pop('model_name', None)
        effective_kwargs.pop('size', None)
        effective_kwargs.pop('model_type', None)
        effective_kwargs.pop('variant', None)
        effective_kwargs.pop('cleanup_model_cache_after_run', None)

        if loader_class == CustomScriptModelLoader:
            logger.debug(f"Instantiating CustomScriptModelLoader for '{model_name}' with effective_kwargs: {list(effective_kwargs.keys())}")
            return loader_class(
                model_name=model_name,
                framework=framework,
                **effective_kwargs
            )
        else:
            if quantization is not None:
                effective_kwargs['quantization'] = quantization
            else:
                pass
            
            effective_kwargs.pop('checkpoint', None)
            logger.debug(f"Instantiating {loader_class.__name__} for '{model_name}' with effective_kwargs: {list(effective_kwargs.keys())}")
            return loader_class(
                model_name=model_name,
                **effective_kwargs
            )