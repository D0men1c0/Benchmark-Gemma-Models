from typing import Any, Optional, Type, Dict
from .base_postprocessor import BasePostProcessor
from .concrete_postprocessors import (
    DefaultPostProcessor, 
    MMLUPostProcessor, 
    GSM8KPostProcessor,
    SummarizationPostProcessor, 
    TranslationPostProcessor,
    GlueSST2OutputPostProcessor,
    GlueMRPCOutputPostProcessor,
    GlueSTSBOutputPostProcessor,
    CreativeTextPostProcessor,
    CustomScriptPostProcessor,
)

from utils.logger import setup_logger

logger = setup_logger(__name__)

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
        "creative_text_output": CreativeTextPostProcessor,
        "custom_script": CustomScriptPostProcessor,
        "generic_generation": DefaultPostProcessor,  # Generic fallback
        "unknown": DefaultPostProcessor,             # Fallback for unknown types
        # Add other task_type -> PostProcessor Class mappings here
    }

    @classmethod
    def register_processor(cls, task_type: str, processor_class: Type[BasePostProcessor]):
        """Dynamically register a new post-processor type."""
        cls._PROCESSOR_REGISTRY[task_type.lower()] = processor_class

    @classmethod
    def get_processor(cls, processor_key: str, processor_options: Optional[Dict[str, Any]] = None) -> BasePostProcessor:
        """
        Gets an instance of the appropriate post-processor.
        'processor_options' can contain 'script_path', 'function_name', 'script_args'
        for the CustomScriptPostProcessor.
        :param processor_key: Key to identify the post-processor type
        :param processor_options: Optional dictionary of options for the post-processor
        :return: An instance of the post-processor
        """
        processor_options = processor_options if processor_options is not None else {}
        # Fallback to DefaultPostProcessor if key not found
        processor_class = cls._PROCESSOR_REGISTRY.get(processor_key.lower())
        if not processor_class:
            logger.warning(f"Post-processor key '{processor_key}' not found in registry. Falling back to 'default'.") # Added logger
            processor_class = cls._PROCESSOR_REGISTRY.get("default", DefaultPostProcessor)


        if processor_class == CustomScriptPostProcessor:
            script_path = processor_options.get("script_path")
            function_name = processor_options.get("function_name")
            script_args = processor_options.get("script_args", {})
            
            # CustomScriptPostProcessor's __init__ will raise an error if one is provided without the other,
            # or will initialize to act like DefaultPostProcessor if both are None.
            logger.debug(f"Instantiating CustomScriptPostProcessor with script: {script_path}, func: {function_name}, args: {script_args}") # Added logger
            return processor_class(script_path=script_path, function_name=function_name, script_args=script_args)
        else:
            # For other processors
            try:
                instance = processor_class()
                # If other post-processors adopt a set_options pattern like metrics:
                if hasattr(instance, 'set_options') and callable(getattr(instance, 'set_options')):
                    # Pass all processor_options; the specific instance will pick what it needs.
                    # This is useful if 'processor_options' contains more than just script_args,
                    # e.g., options for CustomTaskChoicePostProcessor.
                    instance.set_options(**processor_options) 
                    logger.debug(f"Called set_options on {processor_class.__name__} with {processor_options}") # Added logger
                return instance
            except Exception as e: # Catch potential errors during instantiation or set_options
                logger.error(f"Error instantiating or setting options for processor '{processor_key}' (class {processor_class.__name__}): {e}", exc_info=True)
                logger.warning("Falling back to DefaultPostProcessor due to error.")
                return DefaultPostProcessor()