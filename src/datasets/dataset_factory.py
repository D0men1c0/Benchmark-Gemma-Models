from typing import Dict, Any
from .concrete_dataset_loader import ConcreteDatasetLoader

class DatasetFactory:
    """Factory for creating configured dataset loaders"""
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> ConcreteDatasetLoader:
        """Create loader from configuration dictionary"""
        return ConcreteDatasetLoader(
            name=config["name"],
            source_type=config.get("source_type", "hf_hub"),
            config=config.get("config"),
            split=config.get("split", "train"),
            data_dir=config.get("data_dir"),
            streaming=config.get("streaming", False),
            **config.get("loader_args", {})
        )