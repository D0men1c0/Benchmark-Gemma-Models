from typing import Any
from .concrete_task_handlers import ClassificationTaskHandler, GenerationTaskHandler
from .base_task_handler import TaskHandler

class TaskHandlerFactory:
    """Factory to create appropriate task handlers based on task type."""

    _handlers = {
        "classification": ClassificationTaskHandler,
        "generation": GenerationTaskHandler,
    }

    @classmethod
    def get_handler(
        cls, task_type: str, model: Any, tokenizer: Any, device: str
    ) -> TaskHandler:
        """
        Get the appropriate task handler for the given task type.

        :param task_type: Type of task (e.g., 'classification', 'generation').
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :return: Initialized TaskHandler instance.
        """
        handler_class = cls._handlers.get(task_type)
        if not handler_class:
            raise ValueError(f"No handler registered for task type: {task_type}")
        return handler_class(model, tokenizer, device)