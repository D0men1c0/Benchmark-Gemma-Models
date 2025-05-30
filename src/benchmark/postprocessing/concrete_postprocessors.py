import importlib
from pathlib import Path
import re
from typing import Dict, Any, Optional, Tuple, List
from .base_postprocessor import BasePostProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)

# --- Helper Functions (can stay here or move to a shared utils file) ---

# Previous functions not used in the concrete postprocessors
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

def _extract_mmlu_choice_enhanced(generated_text: str, options: str = "ABCD") -> str:
    if not generated_text:
        return "INVALID_PRED"

    text_upper = generated_text.strip().upper()
    valid_options_set = set(list(options))

    explicit_answer_patterns = [
        rf"(?:THE\s+CORRECT\s+ANSWER\s+IS|THE\s+ANSWER\s+IS|ANSWER:|CORRECT\s+OPTION:|OPTION:)\s*\(?([{options}])\)?", # Check for (A), (B), etc.
        rf"CORRECT\s+IS\s+([{options}])", # Check for CORRECT IS A, B, etc.
        rf"IS:\s*([{options}])" # Check for IS: A, B, etc.
    ]

    # Check for explicit answer patterns
    for pattern in explicit_answer_patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)

    formatted_patterns = [
        rf"\(([{options}])\)", # Check for (A), (B), etc.
        rf"\b([{options}])\)", # Check for A), B), etc.
        rf"\b([{options}])\.", # Check for A., B., etc.
        rf"\b([{options}])\b"  # Check for A, B, etc.     
    ]
    
    first_match_position = float('inf')
    best_match_char = None

    # Find the first match in the text
    for pattern in formatted_patterns:
        match = re.search(pattern, text_upper)
        if match and match.start() < first_match_position:
            first_match_position = match.start()
            best_match_char = match.group(1)
    
    if best_match_char:
        return best_match_char

    # Fallback: find the first A,B,C,D character anywhere
    if len(text_upper) == 1 and text_upper in valid_options_set:
        return text_upper
            
    return "INVALID_PRED"

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

# Previous functions not used in the concrete postprocessors
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

def _extract_gsm8k_final_answer_enhanced(generated_text: str) -> str:
    """
    Extracts the final answer from the generated text.
    :param generated_text: The generated text from the model.
    :return: The final number found in the text, cleaned of commas.
             If no valid number is found, returns "ANSWER_NOT_FOUND".
             If the number format is invalid, returns "INVALID_NUM_FORMAT_IN_HASH_PATTERN".
    """
    if not generated_text:
        return "ANSWER_NOT_FOUND"

    text = generated_text.strip()

    matches = list(re.finditer(r"####\s*([\d,.-]+)", text))
    if matches:
        last_match = matches[-1]
        num_str = last_match.group(1).replace(',', '')
        # Refine num_str: if it ends with a period and is not a decimal, remove the period
        if num_str.endswith('.') and '.' not in num_str[:-1]:
            num_str = num_str[:-1]
        try:
            float(num_str)
            return num_str
        except ValueError:
            return "INVALID_NUM_FORMAT_IN_HASH_PATTERN"

    answer_phrases = [
        r"the final answer is",
        r"the answer is",
        r"so the result is",
        r"which gives us",
        r"is equal to"
    ]
    
    best_fallback_num_str = None
    for phrase in answer_phrases:
        phrase_matches = list(re.finditer(rf"{phrase}\s*:?\s*([-+]?\s*[\d,.]+\d)", text, re.IGNORECASE))
        if phrase_matches:
            last_phrase_match = phrase_matches[-1]
            num_str_candidate = last_phrase_match.group(1).replace(',', '').replace(' ', '')
            try:
                float(num_str_candidate)
                best_fallback_num_str = num_str_candidate
                break
            except ValueError:
                continue

    if best_fallback_num_str:
        return best_fallback_num_str

    potential_numbers = re.findall(r"[-+]?\b\d[\d,.]*\b", text)
    if potential_numbers:
        last_num_str = potential_numbers[-1].replace(',', '')
        if last_num_str.endswith('.') and '.' not in last_num_str[:-1]:
            last_num_str = last_num_str[:-1]
        try:
            float(last_num_str)
            return last_num_str
        except ValueError:
            pass

    return "ANSWER_NOT_FOUND"


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
        processed_predictions = [_extract_mmlu_choice_enhanced(text) for text in predictions]
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
        processed_predictions = [_extract_gsm8k_final_answer_enhanced(text) for text in predictions]
        processed_labels = [_extract_gsm8k_final_answer_enhanced(str(label)) for label in labels] # Ensure label is string
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
            # Prioritize "not equivalent" or more specific "no" checks
            if "not equivalent" in text_lower or "no" in text_lower or re.search(r"\bno\b", text_lower):
                processed_predictions.append(0)
            elif "yes" in text_lower or re.search(r"\byes\b", text_lower) or "equivalent" in text_lower:
                processed_predictions.append(1)
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
    

class CreativeTextPostProcessor(BasePostProcessor):
    """
    Post-processor for outputs from creative text generation tasks.
    Example: counts words and removes excessive newlines.
    """
    def process(self, predictions: List[str], labels: List[Any], batch: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Processes creative text predictions.

        :param predictions: Raw generated text from the model for the batch.
        :param labels: Raw labels (referencetexts) from the dataset batch. Può essere una lista di None.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
                 processed_predictions qui sarà una lista di dizionari,
                 ognuno contenente il testo pulito e il conteggio parole.
                 processed_labels rimangono i testi di riferimento.
        """
        processed_predictions_list = []
        for pred_text in predictions:
            if pred_text is None:
                processed_predictions_list.append({"cleaned_text": "", "word_count": 0, "original_text": None})
                continue

            # Start with basic stripping
            cleaned_text = str(pred_text).strip()

            # Normalize line breaks to \n
            cleaned_text = cleaned_text.replace('\r\n', '\n').replace('\r', '\n')

            # Replace multiple newlines with a single special marker
            cleaned_text = re.sub(r'\n\s*\n+', ' %%%PARA_BREAK%%% ', cleaned_text)

            # Replace single newlines (that are not part of marked paragraphs) with a space
            cleaned_text = cleaned_text.replace('\n', ' ')

            # Collapse multiple spaces to a single space
            cleaned_text = " ".join(cleaned_text.split())

            # Restore paragraph breaks and strip leading/trailing spaces around them
            cleaned_text = cleaned_text.replace(' %%%PARA_BREAK%%% ', '\n\n').strip()

            word_count = len(cleaned_text.split())

            processed_predictions_list.append({
                "cleaned_text": cleaned_text,
                "word_count": word_count,
                "original_text": pred_text 
            })

        processed_labels = [str(l) if l is not None else None for l in labels]
        return processed_predictions_list, processed_labels
    
    
class CustomScriptPostProcessor(BasePostProcessor):
    """
    A post-processor that delegates its logic to a user-defined function
    in an external Python script.
    """
    def __init__(self, script_path: Optional[str] = None, 
                 function_name: Optional[str] = None, 
                 script_args: Optional[Dict[str, Any]] = None):
        """
        :param script_path: Path to the Python script containing the custom function.
        :param function_name: Name of the function to call in the script.
        :param script_args: Additional arguments to pass to the custom function.
        """
        super().__init__()
        self.script_path_str = script_path
        self.function_name = function_name
        self.script_args = script_args if script_args is not None else {}
        self.custom_process_fn = None

        if self.script_path_str and self.function_name:
            self.script_path = Path(self.script_path_str)
            if not self.script_path.is_file():
                raise FileNotFoundError(f"Post-processor script not found: {self.script_path}")
            self._load_custom_function()
        elif self.script_path_str or self.function_name: # Only one provided
             raise ValueError("CustomScriptPostProcessor requires both 'script_path' and 'function_name' if one is provided.")
        # If neither is provided, it will act like DefaultPostProcessor unless process is overridden

    def _load_custom_function(self):
        """Loads the custom function from the specified script."""
        try:
            spec = importlib.util.spec_from_file_location(f"custom_postproc_module.{self.function_name}", str(self.script_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for post-processor script {self.script_path}")

            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module) #type: ignore

            if not hasattr(custom_module, str(self.function_name)): # Ensure function_name is str
                raise AttributeError(f"Function '{self.function_name}' not found in script '{self.script_path}'")
            
            self.custom_process_fn = getattr(custom_module, str(self.function_name))
            logger.info(f"CustomScriptPostProcessor loaded {self.function_name} from {self.script_path}")
        except Exception as e:
            logger.error(f"Failed to load custom post-processor function '{self.function_name}' from '{self.script_path}': {e}", exc_info=True)
            raise

    def process(self, predictions: List[Any], labels: List[Any], batch: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], List[Any]]:
        """
        Processes predictions and labels using the custom function.
        :param predictions: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
        """
        if self.custom_process_fn:
            try:
                # The custom function receives predictions, labels, the original batch (for context), and its specific script_args
                # Signature: my_func(predictions, labels, batch, script_args) -> (processed_preds, processed_labels)
                return self.custom_process_fn(
                    predictions=predictions,
                    labels=labels,
                    batch=batch, # Pass the original batch for context if needed
                    script_args=self.script_args
                )
            except Exception as e:
                logger.error(f"Error executing custom post-processor function '{self.function_name}': {e}", exc_info=True)
                # Fallback to default behavior or re-raise
                return DefaultPostProcessor().process(predictions, labels, batch)
        else:
            # Fallback to default if no script specified
            logger.debug("CustomScriptPostProcessor acting as DefaultPostProcessor (no script/function specified).")
            return DefaultPostProcessor().process(predictions, labels, batch)