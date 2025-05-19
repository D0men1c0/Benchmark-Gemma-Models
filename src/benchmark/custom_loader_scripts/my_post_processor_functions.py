import re
from typing import List, Dict, Any, Tuple, Optional

def simple_prefix_remover_postprocessor(
    predictions: List[Any], 
    labels: List[Any], 
    batch: Optional[Dict[str, Any]] = None,
    script_args: Optional[Dict[str, Any]] = None
) -> Tuple[List[Any], List[Any]]:
    """
    A simple custom post-processing function.
    - Removes a specified prefix from string predictions.
    - Converts labels to strings.
    - Passes through non-string predictions and labels unmodified.

    :param predictions: List of model predictions (strings).
    :param labels: List of true labels (can be any type).
    :param batch: Original batch data (optional).
    :param script_args: Additional arguments passed from the YAML configuration.
    :return: A tuple of (processed_predictions, processed_labels).
    """
    if script_args is None:
        script_args = {}

    prefix_to_remove = script_args.get("prefix_to_remove")
    # Example of another arg: whether to make the prefix removal case-insensitive
    # case_insensitive_prefix = script_args.get("case_insensitive_prefix", False)

    processed_predictions = []
    for pred in predictions:
        if isinstance(pred, str) and prefix_to_remove:
            # Simple prefix removal
            # For case-insensitive removal:
            # if case_insensitive_prefix and pred.lower().startswith(prefix_to_remove.lower()):
            #     processed_predictions.append(pred[len(prefix_to_remove):].strip())
            if pred.startswith(prefix_to_remove):
                processed_predictions.append(pred[len(prefix_to_remove):].strip())
            else:
                processed_predictions.append(pred) # No prefix found, keep as is
        else:
            processed_predictions.append(pred) # Not a string or no prefix to remove

    # Simple label processing: ensure they are strings for most text-based metrics
    # More complex label processing could happen here if needed.
    processed_labels = []
    for label in labels:
        if label is not None:
            processed_labels.append(str(label))
        else:
            processed_labels.append(None) # Keep None as None

    return processed_predictions, processed_labels


def advanced_choice_extractor_postprocessor(
    predictions: List[Any],
    labels: List[Any],
    batch: Optional[Dict[str, Any]] = None,
    script_args: Optional[Dict[str, Any]] = None
) -> Tuple[List[Any], List[Any]]:
    """
    An example of a more advanced post-processor for choice-based tasks.
    It uses regex to find choices like "(A)", "A.", "A)", "Choice A:", etc.
    Labels are converted based on a provided map or assumed to be correct.

    :param predictions: List of model predictions (strings).
    :param labels: List of true labels (can be any type).
    :param batch: Original batch data (optional).
    :param script_args: Additional arguments passed from the YAML configuration.
    :return: A tuple of (processed_predictions, processed_labels).
    """
    if script_args is None:
        script_args = {}

    valid_choices = script_args.get("valid_choices", ["A", "B", "C", "D"]) # Default MMLU-like
    label_map = script_args.get("label_map")
    label_is_direct_choice = script_args.get("label_is_direct_choice", False)
    default_invalid_pred = script_args.get("default_invalid_prediction", "INVALID_PRED")
    default_invalid_label = script_args.get("default_invalid_label", "INVALID_LABEL")

    processed_predictions = []
    for pred_text in predictions:
        if not isinstance(pred_text, str):
            processed_predictions.append(default_invalid_pred)
            continue

        cleaned_text = pred_text.strip().upper()
        found_choice = None

        for choice_char in valid_choices:
            # Regex patterns to find choices like (A), A., A), Choice A, etc.
            # Ensure choice_char is escaped if it can contain regex special characters
            escaped_choice = re.escape(choice_char.upper())
            patterns = [
                r'\(\s*' + escaped_choice + r'\s*\)',  # (A), ( A )
                r'\b' + escaped_choice + r'\s*\.',     # A.
                r'\b' + escaped_choice + r'\s*\)',     # A)
                r'\b' + escaped_choice + r'\b',        # A (as a whole word)
                r'choice\s*' + escaped_choice + r'[:\s]',# Choice A: or Choice A 
                r'option\s*' + escaped_choice + r'[:\s]' # Option A: or Option A
            ]
            for pattern in patterns:
                match = re.search(pattern, cleaned_text)
                if match:
                    found_choice = choice_char # Return the original case version from valid_choices
                    break
            if found_choice:
                break
        
        if found_choice:
            processed_predictions.append(found_choice)
        else:
            # Fallback: if no pattern matched, check if any of the choice characters are present
            # This is a simpler check if the above regex is too strict or misses a format
            for choice_char in valid_choices:
                if choice_char.upper() in cleaned_text: # Check against the cleaned_text
                    found_choice = choice_char
                    break
            processed_predictions.append(found_choice if found_choice else default_invalid_pred)

    # Process labels
    processed_labels = []
    for label_val in labels:
        if label_is_direct_choice:
            str_label = str(label_val)
            # Check if the label is one of the valid choices (case-insensitive for robustness)
            if any(str_label.upper() == vc.upper() for vc in valid_choices):
                # Find the original case version from valid_choices
                original_case_label = next((vc for vc in valid_choices if vc.upper() == str_label.upper()), default_invalid_label)
                processed_labels.append(original_case_label)
            else:
                processed_labels.append(default_invalid_label)
        elif label_map:
            try:
                # Attempt to use label_val as a key, convert to int if it looks like an int
                key = label_val
                if isinstance(label_val, str) and label_val.isdigit(): key = int(label_val)
                elif isinstance(label_val, float) and label_val.is_integer(): key = int(label_val)
                
                processed_labels.append(label_map.get(key, default_invalid_label))
            except (ValueError, TypeError):
                processed_labels.append(default_invalid_label)
        else: # Assume label is already in the correct string format
            processed_labels.append(str(label_val) if label_val is not None else default_invalid_label)
            
    return processed_predictions, processed_labels