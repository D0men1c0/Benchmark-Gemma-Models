from typing import Dict, Any, List, Union, Optional
from utils.logger import setup_logger 
logger = setup_logger(__name__)

# --- Example 1: Custom Metric - Average Prediction Length ---

def init_avg_pred_length_state(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize the state for the average prediction length metric.
    :param options: Optional dictionary of options for the metric.
    :return: A dictionary to hold the state.
    """
    logger.debug(f"CustomMetric (AvgPredLength): Initializing state with options: {options}")
    return {"total_word_count": 0, "num_predictions": 0}

def update_avg_pred_length_state(
    current_state: Dict[str, Any], 
    predictions: List[Any], 
    labels: List[Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update the state with the predictions and labels.
    :param current_state: The current state of the metric.
    :param predictions: The predictions made by the model.
    :param labels: The true labels for the predictions.
    :param options: Optional dictionary of options for the metric.
    :return: The updated state.
    """
    for pred in predictions:
        if isinstance(pred, str):
            current_state["total_word_count"] += len(pred.split())
            current_state["num_predictions"] += 1
        elif isinstance(pred, dict) and "cleaned_text" in pred:
            current_state["total_word_count"] += len(pred["cleaned_text"].split())
            current_state["num_predictions"] += 1
    return current_state

def result_avg_pred_length(
    final_state: Dict[str, Any], 
    options: Optional[Dict[str, Any]] = None
) -> Union[float, Dict[str, float]]:
    """
    Calculate the average prediction length from the final state.
    :param final_state: The final state of the metric.
    :param options: Optional dictionary of options for the metric.
    :return: The average prediction length or a dictionary with the result.
    """
    if final_state["num_predictions"] == 0:
        return {"avg_prediction_length": 0.0}
    
    avg_length = final_state["total_word_count"] / final_state["num_predictions"]
    
    result_key_name = options.get("result_key_name", "avg_prediction_length") if options else "avg_prediction_length"
    logger.debug(f"CustomMetric (AvgPredLength): Final avg length: {avg_length} using key '{result_key_name}'")
    return {result_key_name: avg_length}

# --- Example 2: Custom Metric - Keyword Match Count ---

def init_keyword_match_state(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize the state for the keyword match metric.
    :param options: Optional dictionary of options for the metric.
    :return: A dictionary to hold the state.
    """
    logger.debug(f"CustomMetric (KeywordMatch): Initializing state with options: {options}")
    keywords = []
    if options and "keywords_to_match" in options:
        keywords = options["keywords_to_match"]
        if not isinstance(keywords, list):
            logger.warning("CustomMetric (KeywordMatch): 'keywords_to_match' should be a list. Using empty list.")
            keywords = []
    else:
        logger.warning("CustomMetric (KeywordMatch): 'keywords_to_match' not found in options. Metric may not work as expected.")
    
    return {"match_count": 0, "total_predictions": 0, "keywords": [str(k).lower() for k in keywords]}

def update_keyword_match_state(
    current_state: Dict[str, Any], 
    predictions: List[Any], 
    labels: List[Any], 
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update the state with the predictions and labels.
    :param current_state: The current state of the metric.
    :param predictions: The predictions made by the model.
    :param labels: The true labels for the predictions.
    :param options: Optional dictionary of options for the metric.
    :return: The updated state.
    """
    keywords = current_state.get("keywords", [])
    if not keywords:
        current_state["total_predictions"] += len(predictions)
        return current_state

    for pred in predictions:
        current_state["total_predictions"] += 1
        text_to_search = ""
        if isinstance(pred, str):
            text_to_search = pred.lower()
        elif isinstance(pred, dict) and "cleaned_text" in pred:
            text_to_search = pred["cleaned_text"].lower()
        
        if text_to_search:
            for keyword in keywords:
                if keyword in text_to_search:
                    current_state["match_count"] += 1
                    break 
    return current_state

def result_keyword_match(
    final_state: Dict[str, Any], 
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Calculate the keyword match ratio from the final state.
    :param final_state: The final state of the metric.
    :param options: Optional dictionary of options for the metric.
    :return: A dictionary with the keyword match count and ratio.
    """
    match_count = final_state.get("match_count", 0)
    total_predictions = final_state.get("total_predictions", 0)
    
    match_ratio = 0.0
    if total_predictions > 0:
        match_ratio = match_count / total_predictions
    
    logger.debug(f"CustomMetric (KeywordMatch): Final match_count: {match_count}, ratio: {match_ratio}")
    return {
        "keyword_match_count": float(match_count),
        "keyword_match_ratio": match_ratio
    }