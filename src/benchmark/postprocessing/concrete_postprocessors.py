import re
from typing import Dict, Any, Tuple, List
from .base_postprocessor import BasePostProcessor

# --- Helper Functions (can stay here or move to a shared utils file) ---

def _extract_mmlu_prediction(generated_text: str) -> str:
    """
    Extracts the first valid A, B, C, or D letter from the prediction.
    This is used for MMLU tasks where the model generates a letter as the answer.
    :param generated_text: The generated text from the model.
    :return: The first valid letter (A, B, C, or D) found in the generated text.
             If none found, returns "INVALID_PRED".
    """
    clean_text = generated_text.strip().upper()
    match = re.search(r'\b([A-D])\b', clean_text) # Look for A,B,C,D as a "word"
    if match:
        return match.group(1)
    # Fallback: find the first A,B,C,D character anywhere
    for char in clean_text:
        if char in ['A', 'B', 'C', 'D']:
            return char
    return "INVALID_PRED" # Default if nothing is found

def _convert_mmlu_label(label_index: Any) -> str:
    """
    Converts MMLU index (0-3) to the corresponding letter.
    :param label_index: The index of the label (0-3).
    :return: The corresponding letter (A, B, C, or D).
             If index is invalid, returns "INVALID_LABEL".
    """
    index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    try:
        return index_to_letter.get(int(label_index), "INVALID_LABEL")
    except (ValueError, TypeError):
        return "INVALID_LABEL"

def _extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final numerical answer from a GSM8K string.
    :param text: The generated text from the model.
    :return: The final number found in the text, cleaned of commas.
             If no valid number is found, returns "ANSWER_NOT_FOUND".
             If the number format is invalid, returns "INVALID_NUM_FORMAT".
    """
    # Search for the '#### <number>' pattern at the end of the string
    match = re.search(r"####\s*([\d,.-]+)$", text)
    if match:
        # Clean the number string (remove commas)
        num_str = match.group(1).replace(',', '')
        try:
            # Try converting to float to validate, but return as string for exact_match
            float(num_str)
            return num_str
        except ValueError:
            return "INVALID_NUM_FORMAT"
    return "ANSWER_NOT_FOUND" # Default if pattern is not found

# --- Concrete Classes ---

class DefaultPostProcessor(BasePostProcessor):
    """Default post-processor: simply converts labels to strings."""
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[str], List[str]]:
        """
        Default processing: converts labels to strings.
        :param predictions: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (predictions, labels).
                 Predictions remain unchanged, labels are converted to strings.
        """
        # Ensure labels are strings, does nothing to predictions
        processed_labels = [str(l) for l in labels]
        return predictions, processed_labels

class MMLUPostProcessor(BasePostProcessor):
    """Post-processor for MMLU (letter generation task)."""
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[str], List[str]]:
        """
        Processes MMLU predictions and labels.
        :param predictions: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional, might be needed for context).
        :return: Tuple of (processed_predictions, processed_labels).
                    Predictions are cleaned to extract the first valid letter (A, B, C, or D).
        """
        processed_predictions = [_extract_mmlu_prediction(text) for text in predictions]
        processed_labels = [_convert_mmlu_label(label_idx) for label_idx in labels]
        return processed_predictions, processed_labels

class GSM8KPostProcessor(BasePostProcessor):
    """Post-processor for GSM8K specifically for Exact Match of the final answer."""
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[str], List[str]]:
        """
        Processes GSM8K predictions and labels.
        :param predictions: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
                    Predictions are cleaned to extract the final number.
        """
        # Extracts only the final number from both predictions and labels
        processed_predictions = [_extract_gsm8k_answer(text) for text in predictions]
        processed_labels = [_extract_gsm8k_answer(str(label)) for label in labels] # Ensure label is string
        return processed_predictions, processed_labels

# Add other post-processors here if needed (e.g., for Summarization, Translation if specific cleaning is required)
# Example: Inherit from Default if only basic cleaning is needed
class SummarizationPostProcessor(DefaultPostProcessor):
    pass
class TranslationPostProcessor(DefaultPostProcessor):
    pass