from typing import Dict, Type, Optional, Any
from .base_prompt_builder import BasePromptBuilder
from .concrete_prompt_builders import (
    TemplateBasedPromptBuilder,
    MMLUPromptBuilder,
    TranslationPromptBuilder,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PromptBuilderFactory:
    """
    Factory to create appropriate prompt builders.
    """
    _BUILDER_REGISTRY: Dict[str, Type[BasePromptBuilder]] = {
        "default": TemplateBasedPromptBuilder, # A generic fallback
        "template_based": TemplateBasedPromptBuilder,
        "mmlu": MMLUPromptBuilder,
        "translation": TranslationPromptBuilder,
        # Add other prompt_builder_types and their corresponding classes
    }

    @classmethod
    def register_builder(cls, builder_type: str, builder_class: Type[BasePromptBuilder]):
        cls._BUILDER_REGISTRY[builder_type.lower()] = builder_class

    @classmethod
    def get_builder(
        cls,
        builder_type: Optional[str] = None, # Type of builder requested
        prompt_template: Optional[str] = None, # The actual template string
        handler_args: Optional[Dict[str, Any]] = None # Other args like lang codes
    ) -> BasePromptBuilder:
        """
        Get an initialized prompt builder.

        :param builder_type: Type of prompt builder to get (e.g., "mmlu", "translation").
                             If None, "default" (TemplateBasedPromptBuilder) is used.
        :param prompt_template: The template string to be used by the builder.
        :param handler_args: Additional arguments for initializing the builder (e.g., lang codes).
        :return: An initialized instance of a BasePromptBuilder subclass.
        """
        builder_type_to_use = (builder_type or "default").lower()
        
        builder_class = cls._BUILDER_REGISTRY.get(builder_type_to_use)
        if not builder_class:
            logger.warning(f"No prompt builder registered for type '{builder_type_to_use}'. Falling back to TemplateBasedPromptBuilder.")
            builder_class = TemplateBasedPromptBuilder

        return builder_class(template_string=prompt_template, handler_args=handler_args)