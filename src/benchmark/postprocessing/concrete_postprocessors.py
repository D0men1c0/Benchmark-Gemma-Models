import re
from typing import Dict, Any, Tuple, List
from .base_postprocessor import BasePostProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)

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

class SummarizationPostProcessor(DefaultPostProcessor):
    # No specific processing needed for summarization
    pass

class TranslationPostProcessor(DefaultPostProcessor):
    pass


class GlueSST2OutputPostProcessor(BasePostProcessor):
    """
    Post-processor for GLUE SST-2 sentiment classification outputs from a prompted LLM.
    Maps textual sentiment predictions (e.g., "positive", "negative") to class indices (0 or 1).
    """
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[int], List[int]]:
        """
        Processes raw LLM outputs for SST-2.

        :param predictions: List of raw generated text from the model.
        :param labels: List of original labels (expected to be 0 or 1).
        :param batch: The original batch data (optional).
        :return: Tuple of (list_of_predicted_class_indices, list_of_original_class_indices).
                 SST-2 labels: 0 for negative, 1 for positive.
        """
        processed_predictions: List[int] = []
        for pred_text in predictions:
            text_lower = pred_text.strip().lower()
            # More robust parsing can be added here
            if "positive" in text_lower or re.search(r"\bpos\b", text_lower):
                processed_predictions.append(1)
            elif "negative" in text_lower or re.search(r"\bneg\b", text_lower):
                processed_predictions.append(0)
            else:
                logger.warning(f"SST-2 PostProcessor: Could not reliably parse sentiment from '{pred_text}'. Defaulting to 0 (invalid).")
                processed_predictions.append(0) # Invalid prediction marker

        # Ensure labels are integers
        processed_labels = [int(l) if isinstance(l, (int, float, str)) and str(l).isdigit() else -1 for l in labels]
        return processed_predictions, processed_labels


class GlueMRPCOutputPostProcessor(BasePostProcessor):
    """
    Post-processor for GLUE MRPC paraphrase detection outputs from a prompted LLM.
    Maps textual predictions (e.g., "yes", "no") to class indices (0 or 1).
    """
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[int], List[int]]:
        """
        Processes raw LLM outputs for MRPC.

        :param predictions: List of raw generated text from the model.
        :param labels: List of original labels (expected to be 0 or 1).
        :param batch: The original batch data (optional).
        :return: Tuple of (list_of_predicted_class_indices, list_of_original_class_indices).
                 MRPC labels: 0 for not equivalent, 1 for equivalent.
        """
        processed_predictions: List[int] = []
        for pred_text in predictions:
            text_lower = pred_text.strip().lower()
            if "yes" in text_lower or re.search(r"\byes\b", text_lower) or "equivalent" in text_lower:
                processed_predictions.append(1)
            elif "no" in text_lower or re.search(r"\bno\b", text_lower) or "not equivalent" in text_lower:
                processed_predictions.append(0)
            else:
                logger.warning(f"MRPC PostProcessor: Could not reliably parse paraphrase judgment from '{pred_text}'. Defaulting to class 0.")
                processed_predictions.append(0) # Invalid prediction marker

        processed_labels = [int(l) if isinstance(l, (int, float, str)) and str(l).isdigit() else -1 for l in labels]
        return processed_predictions, processed_labels


class GlueSTSBOutputPostProcessor(BasePostProcessor):
    """
    Post-processor for GLUE STS-B semantic similarity outputs from a prompted LLM.
    Extracts a float score from the model's textual output.
    """
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[float], List[float]]:
        """
        Processes raw LLM outputs for STS-B.

        :param predictions: List of raw generated text from the model.
        :param labels: List of original labels (expected to be float scores).
        :param batch: The original batch data (optional).
        :return: Tuple of (list_of_predicted_scores, list_of_original_scores).
        """
        processed_predictions: List[float] = []
        default_invalid_score = -1.0

        for pred_text in predictions:
            text_to_search = pred_text.strip()
            # Regex to find numbers, including decimals.
            # More robust parsing might be needed depending on LLM output format.
            match = re.search(r"(\d+\.?\d*|\.\d+)", text_to_search)
            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to 0-5 range for STS-B, or normalize if needed
                    score = max(0.0, min(5.0, score))
                    processed_predictions.append(score)
                except ValueError:
                    logger.warning(f"STSB PostProcessor: Found number but failed to convert to float: '{match.group(1)}' in '{pred_text}'. Defaulting.")
                    processed_predictions.append(default_invalid_score)
            else:
                logger.warning(f"STSB PostProcessor: Could not parse similarity score from '{pred_text}'. Defaulting.")
                processed_predictions.append(default_invalid_score)

        processed_labels = []
        for l in labels:
            try:
                processed_labels.append(float(l))
            except (ValueError, TypeError):
                logger.warning(f"STSB PostProcessor: Could not convert label '{l}' to float. Defaulting.")
                processed_labels.append(default_invalid_score)
        return processed_predictions, processed_labels