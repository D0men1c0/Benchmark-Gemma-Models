from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional

class BaseDatasetLoader(ABC):
    """Abstract interface for dataset loaders"""
    
    @abstractmethod
    def __init__(
        self,
        name: str,
        source_type: str = "hf_hub",
        config: Optional[str] = None,
        split: str = "train",
        data_dir: Optional[str] = None,
        streaming: bool = False,
        **loader_kwargs
    ):
        pass
    
    @abstractmethod
    def load(self) -> Iterable:
        """Load dataset from configured source"""
        pass
    
    @classmethod
    @abstractmethod
    def register_source(cls, source_type: str, loader_fn: Callable):
        """Register custom data source"""
        pass