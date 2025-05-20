from typing import Dict, Any
from .base_dataset_loader import BaseDatasetLoader
from .concrete_dataset_loader import ConcreteDatasetLoader, CustomScriptDatasetLoader

from utils.logger import setup_logger

logger = setup_logger(__name__)

class DatasetFactory:
    """
    Factory class to create dataset loaders using a registry pattern.
    This class allows for dynamic registration of new dataset loader types.
    The factory method `from_config` will instantiate the appropriate dataset loader
    based on the provided configuration.
    """

    _LOADER_CLASSES: Dict[str, type[BaseDatasetLoader]] = {
        "hf_hub": ConcreteDatasetLoader,
        "local": ConcreteDatasetLoader,
        "custom_script": CustomScriptDatasetLoader,
    }

    @classmethod
    def register_loader_class(cls, source_type: str, loader_class: type[BaseDatasetLoader]):
        cls._LOADER_CLASSES[source_type.lower()] = loader_class

    @staticmethod
    def from_config(config: Dict[str, Any]) -> BaseDatasetLoader:
        """
        Create a dataset loader instance based on the provided configuration.
        :param config: Dictionary containing dataset configuration.
        :return: An instance of a dataset loader.
        """
        source_type_from_config = config.get("source_type", "hf_hub")
        source_type_lower = source_type_from_config.lower()
        loader_class = DatasetFactory._LOADER_CLASSES.get(source_type_lower)

        if not loader_class:
            raise ValueError(f"Unsupported dataset source_type: '{source_type_from_config}'. "
                             f"Registered types: {list(DatasetFactory._LOADER_CLASSES.keys())}")

        kwargs_for_constructor = config.copy()
        dataset_name = kwargs_for_constructor.pop("name")
        kwargs_for_constructor.pop("source_type", None)

        logger.debug(
            f"Instantiating {loader_class.__name__} for dataset '{dataset_name}' "
            f"(source_type: '{source_type_from_config}') with kwargs: {kwargs_for_constructor}"
        )
        
        return loader_class(
            name=dataset_name, 
            source_type=source_type_from_config, 
            **kwargs_for_constructor
        )