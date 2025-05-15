from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BasePromptBuilder(ABC):
    """
    Abstract base class for prompt building strategies.
    Each concrete builder will implement how to construct prompts
    for a specific style or task structure.
    """

    def __init__(self, template_string: Optional[str] = None, handler_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt builder.

        :param template_string: The main prompt template string.
        :param handler_args: Additional arguments from the task handler or task config,
                               which might include other template parts or specific settings
                               (e.g., few-shot example format, language codes).
        """
        self.template_string = template_string
        self.handler_args = handler_args if handler_args else {}

    @abstractmethod
    def build_prompts(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Builds a list of prompts from a list of batch items.
        Each item in batch_items is a dictionary representing one instance from the dataset.

        :param batch_items: A list of dictionaries, where each dictionary
                            contains the necessary data fields for one prompt
                            (e.g., {"question": "...", "choices_formatted_str": "..."}).
        :return: A list of fully formatted prompt strings.
        """
        pass

    def _format_single_prompt(self, template: str, data: Dict[str, Any], fallback_text: str = "") -> str:
        """
        Helper method to safely format a single prompt string.
        Can be used by concrete implementations.

        :param template: The prompt template string.
        :param data: Dictionary of data to fill into the template.
        :param fallback_text: Text to return if formatting fails.
        :return: Formatted prompt string or fallback.
        """
        try:
            return template.format(**data)
        except KeyError as e:
            # self.logger.error(f"Prompt formatting missing key: {e}. Template: '{template}'. Data keys: {list(data.keys())}")
            print(f"Warning: Prompt formatting missing key: {e}. Template: '{template}'. Data keys: {list(data.keys())}") # Placeholder for logger
            return fallback_text # Or re-raise, or use a more specific fallback from data
        except Exception as ex:
            # self.logger.error(f"Generic prompt formatting error: {ex}. Template: '{template}'")
            print(f"Warning: Generic prompt formatting error: {ex}. Template: '{template}'") # Placeholder for logger
            return fallback_text
        return fallback_text # Should not be reached if try succeeds