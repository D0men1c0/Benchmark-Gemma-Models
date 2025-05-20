from typing import Any, Dict, Type
from .base_task_handler import TaskHandler
from .concrete_task_handlers import (
    ClassificationTaskHandler,
    SummarizationTaskHandler,
    TranslationTaskHandler,
    MultipleChoiceQATaskHandler,
    MathReasoningGenerationTaskHandler,
    TextPairClassificationTaskHandler,
    GlueClassificationPromptingTaskHandler,
    GlueTextPairPromptingTaskHandler,
    CreativeTextGenerationTaskHandler,
    CustomScriptTaskHandler,
)

class TaskHandlerFactory:
    """Factory to create appropriate task handlers based on task type."""

    _handlers: Dict[str, Type[TaskHandler]] = {
        "classification": ClassificationTaskHandler, # For pretrained logits models
        "summarization": SummarizationTaskHandler,
        "translation": TranslationTaskHandler,
        "multiple_choice_qa": MultipleChoiceQATaskHandler,
        "math_reasoning_generation": MathReasoningGenerationTaskHandler,
        "text_pair_classification": TextPairClassificationTaskHandler, # For pretrained logits models
        "glue_classification_prompting": GlueClassificationPromptingTaskHandler,
        "glue_text_pair_prompting": GlueTextPairPromptingTaskHandler,
        "creative_text_generation": CreativeTextGenerationTaskHandler,
        "custom_script": CustomScriptTaskHandler,
    }

    @classmethod
    def register_handler(cls, task_type: str, handler_class: Type[TaskHandler]):
        """
        Register a new task handler.
        :param task_type: Type of task (e.g., 'classification', 'generation').
        :param handler_class: Class of the task handler to register.
        :raises ValueError: If the task type is already registered.
        """
        cls._handlers[task_type.lower()] = handler_class

    @classmethod
    def get_handler(
        cls, task_type: str, model: Any, tokenizer: Any, device: str, advanced_args: Dict[str, Any] = None
    ) -> TaskHandler:
        """
        Get the appropriate task handler for the given task type.

        :param task_type: Type of task (e.g., 'classification', 'generation').
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Additional arguments for the task handler.
        :return: Initialized TaskHandler instance.
        """
        handler_class = cls._handlers.get(task_type.lower())
        if not handler_class:
            raise ValueError(f"No handler registered for task type: {task_type}")

        return handler_class(model, tokenizer, device, advanced_args or {})