from typing import Type, Dict
from .base_postprocessor import BasePostProcessor
from .concrete_postprocessors import (
    DefaultPostProcessor, 
    MMLUPostProcessor, 
    GSM8KPostProcessor,
    SummarizationPostProcessor, 
    TranslationPostProcessor,
    GlueSST2OutputPostProcessor,
    GlueMRPCOutputPostProcessor,
    GlueSTSBOutputPostProcessor
)

class PostProcessorFactory:
    """Factory to create task-specific post-processor instances."""

    _PROCESSOR_REGISTRY: Dict[str, Type[BasePostProcessor]] = {
        "summarization": SummarizationPostProcessor, # Use default or specific SummarizationPostProcessor
        "translation": TranslationPostProcessor,     # Use default or specific TranslationPostProcessor
        "mmlu_generation": MMLUPostProcessor,        # Use specific MMLU processor
        "gsm8k": GSM8KPostProcessor,                 # Use specific GSM8K processor (for Exact Match)
        "glue_sst2_output": GlueSST2OutputPostProcessor,
        "glue_mrpc_output": GlueMRPCOutputPostProcessor,
        "glue_stsb_output": GlueSTSBOutputPostProcessor,
        "generic_generation": DefaultPostProcessor,  # Generic fallback
        "unknown": DefaultPostProcessor,             # Fallback for unknown types
        # Add other task_type -> PostProcessor Class mappings here
    }

    @classmethod
    def register_processor(cls, task_type: str, processor_class: Type[BasePostProcessor]):
        """Dynamically register a new post-processor type."""
        cls._PROCESSOR_REGISTRY[task_type.lower()] = processor_class

    @classmethod
    def get_processor(cls, task_type: str) -> BasePostProcessor:
        """Gets an instance of the appropriate post-processor for the task_type."""
        processor_class = cls._PROCESSOR_REGISTRY.get(task_type.lower(), DefaultPostProcessor)
        return processor_class() # Return an instance of the class