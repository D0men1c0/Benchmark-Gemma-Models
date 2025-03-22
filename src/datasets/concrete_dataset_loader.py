from typing import Dict, Any, Iterable, Optional, Callable
from datasets import load_dataset as hf_load_dataset
import os
import logging
from .base_dataset_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)

class ConcreteDatasetLoader(BaseDatasetLoader):
    """Concrete implementation of dataset loader"""
    
    _SOURCE_REGISTRY: Dict[str, Callable] = {}
    
    def __init__(self, name: str, source_type: str = "hf_hub", 
                config: Optional[str] = None, split: str = "train",
                data_dir: Optional[str] = None, streaming: bool = False, 
                **loader_kwargs):
        
        # Validations
        if not isinstance(name, str) or len(name.strip()) == 0:
            raise ValueError("Invalid dataset name")
            
        valid_splits = ["train", "validation", "test", 
                       "train+validation", "all"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Valid: {valid_splits}")
        
        # Initialization
        self.name = name
        self.source_type = source_type
        self.config = config
        self.split = split
        self.data_dir = data_dir
        self.streaming = streaming
        self.loader_kwargs = loader_kwargs

    @classmethod
    def register_source(cls, source_type: str, loader_fn: Callable):
        """
        Register a custom data source loader function.

        :param source_type: Source type identifier
        :param loader_fn: Loader function to register
        """
        cls._SOURCE_REGISTRY[source_type.lower()] = loader_fn

    def load(self) -> Iterable:
        """"Load the dataset from the configured source."""
        source_type = self.source_type.lower()
        
        if source_type in self._SOURCE_REGISTRY:
            return self._load_custom(source_type)
            
        if source_type == "hf_hub":
            return self._load_hf_dataset()
        if source_type == "local":
            return self._load_local_files()
            
        raise ValueError(f"Unsupported source: {source_type}")

    def _load_hf_dataset(self):
        """Load dataset from Hugging Face Hub."""
        return hf_load_dataset(
            self.name,
            self.config,
            split=self.split,
            streaming=self.streaming,
            **self.loader_kwargs
        )

    def _load_local_files(self):
        """Load dataset from local files."""
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise ValueError(f"Invalid data dir: {self.data_dir}")
            
        valid_formats = ["csv", "json", "parquet", "text"]
        if self.name.lower() not in valid_formats:
            raise ValueError(f"Unsupported format: {self.name}")
            
        return hf_load_dataset(
            self.name,
            data_dir=self.data_dir,
            split=self.split,
            streaming=self.streaming,
            **self.loader_kwargs
        )

    def _load_custom(self, source_type: str):
        """Load dataset using a custom loader."""
        return self._SOURCE_REGISTRY[source_type](
            name=self.name,
            config=self.config,
            split=self.split,
            data_dir=self.data_dir,
            streaming=self.streaming,
            **self.loader_kwargs
        )