from abc import abstractmethod
from typing import Any, List, Union, Dict, Optional # Added Optional
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from rouge import Rouge
import bert_score
from collections import Counter
import math
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report as seqeval_classification_report
from transformers import pipeline
from .base_metrics import BaseMetric
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AccuracyMetric(BaseMetric):
    """ Computes accuracy as the ratio of correct predictions to total predictions."""

    def __init__(self):
        """Initializes the AccuracyMetric instance."""
        super().__init__()
        self.correct_predictions: int = 0
        self.total_predictions: int = 0

    def reset_state(self) -> None:
        """ Resets the internal state of the metric."""
        self.correct_predictions = 0
        self.total_predictions = 0

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        for pred, label in zip(predictions, labels):
            if pred == label:
                self.correct_predictions += 1
            self.total_predictions += 1

    def result(self) -> float:
        """
        Computes and returns the final accuracy value.
        :return: Computed accuracy value.
        """
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


class SklearnScoreMetric(BaseMetric):
    """Base class for metrics that use sklearn scoring functions."""

    def __init__(self):
        """Initializes the SklearnScoreMetric instance."""
        super().__init__()
        self._collected_predictions: List[Any] = []
        self._collected_labels: List[Any] = []

    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self._collected_predictions = []
        self._collected_labels = []

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        self._collected_predictions.extend(predictions)
        self._collected_labels.extend(labels)

    def _compute_score(self, score_func, **kwargs) -> float:
        """
        Computes the score using the provided sklearn scoring function.
        :param score_func: The sklearn scoring function to use.
        :param kwargs: Additional keyword arguments for the scoring function.
        :return: Computed score.
        """
        if not self._collected_predictions or not self._collected_labels:
            logger.warning(f"{self.__class__.__name__}: No data collected to compute score.")
            return 0.0
        if len(self._collected_predictions) != len(self._collected_labels):
            logger.error(f"{self.__class__.__name__}: Predictions and labels length mismatch. Cannot compute score.")
            return 0.0

        sklearn_options = {
            "average": self._options.get("average", "weighted"), # Default 'weighted' for multi-class
            "zero_division": self._options.get("zero_division", 0) # Default 0 to avoid warnings/errors
        }
        
        try:
            return score_func(self._collected_labels, self._collected_predictions, **sklearn_options)
        except ValueError as ve:
            logger.error(f"ValueError computing {self.__class__.__name__} with sklearn: {ve}. Options: {sklearn_options}")
            return 0.0
        except Exception as e:
            logger.error(f"Error computing {self.__class__.__name__} with sklearn: {e}")
            return 0.0


class PrecisionMetric(SklearnScoreMetric):
    """ Computes precision score using sklearn's precision_score function. """
    def result(self) -> float:
        """ Computes and returns the precision score. """
        return self._compute_score(precision_score)

class RecallMetric(SklearnScoreMetric):
    """ Computes recall score using sklearn's recall_score function. """
    def result(self) -> float:
        """ Computes and returns the recall score. """
        return self._compute_score(recall_score)

class F1ScoreMetric(SklearnScoreMetric):
    """ Computes F1 score using sklearn's f1_score function. """
    def result(self) -> float:
        """ Computes and returns the F1 score. """
        return self._compute_score(f1_score)


class ExactMatchMetric(BaseMetric):
    """ Computes the exact match ratio between predictions and labels. """
    def __init__(self):
        """Initializes the ExactMatchMetric instance."""
        super().__init__()
        self.match_count: int = 0
        self.total_count: int = 0

    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self.match_count = 0
        self.total_count = 0

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        normalize = self._options.get("normalize", False)
        ignore_case = self._options.get("ignore_case", False)
        ignore_punct = self._options.get("ignore_punct", False)

        for pred_item, label_item in zip(predictions, labels):
            pred_str = str(pred_item)
            label_str = str(label_item)

            if normalize:
                if ignore_case:
                    pred_str = pred_str.lower()
                    label_str = label_str.lower()
                if ignore_punct:
                    # Basic punctuation removal. For more robust removal, consider regex or a dedicated library.
                    punct_to_remove = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
                    pred_str = ''.join(c for c in pred_str if c not in punct_to_remove).strip()
                    label_str = ''.join(c for c in label_str if c not in punct_to_remove).strip()
                    # Also consider normalizing whitespace if ignore_punct is true
                    pred_str = ' '.join(pred_str.split())
                    label_str = ' '.join(label_str.split())


            if pred_str == label_str:
                self.match_count += 1
            self.total_count += 1

    def result(self) -> float:
        """ Computes and returns the exact match ratio. """
        if self.total_count == 0:
            return 0.0
        return self.match_count / self.total_count


class BLEUScoreMetric(BaseMetric):
    """ Computes BLEU score using NLTK's corpus_bleu function. """

    def __init__(self):
        """Initializes the BLEUScoreMetric instance."""
        super().__init__()
        self._collected_predictions_tokens: List[List[str]] = []
        self._collected_labels_tokens: List[List[List[str]]] = []

    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self._collected_predictions_tokens = []
        self._collected_labels_tokens = []

    def update_state(self, predictions: List[str], labels: List[Union[str, List[str]]]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        for pred_text, label_item in zip(predictions, labels):
            # Ensure pred_text is a string before splitting
            if not isinstance(pred_text, str):
                logger.warning(f"BLEU: Prediction is not a string ({type(pred_text)}), skipping item.")
                continue
            self._collected_predictions_tokens.append(pred_text.split())

            if isinstance(label_item, str):
                self._collected_labels_tokens.append([label_item.split()])
            elif isinstance(label_item, list) and all(isinstance(ref, str) for ref in label_item):
                self._collected_labels_tokens.append([ref_text.split() for ref_text in label_item])
            else:
                logger.warning(f"BLEU: Unsupported label format ({type(label_item)}), skipping item.")

    def result(self) -> float:
        """ Computes and returns the BLEU score. """
        if not self._collected_predictions_tokens or not self._collected_labels_tokens:
            return 0.0

        weights = self._options.get("weights", (0.25, 0.25, 0.25, 0.25)) # Default BLEU-4
        smoothing_option_config = self._options.get("smoothing", "method1") # Default to method1

        smoothing_function_map = {
            "method0": SmoothingFunction().method0, # No smoothing
            "method1": SmoothingFunction().method1,
            "method2": SmoothingFunction().method2,
            "method3": SmoothingFunction().method3,
            "method4": SmoothingFunction().method4,
            "method5": SmoothingFunction().method5,
            "method6": SmoothingFunction().method6,
            "method7": SmoothingFunction().method7,
        }
        smoothing_function = smoothing_function_map.get(smoothing_option_config)
        if smoothing_option_config is True: # Backward compatibility if True was passed
            smoothing_function = SmoothingFunction().method1
        elif smoothing_option_config is False or smoothing_option_config is None:
             smoothing_function = SmoothingFunction().method0 # No smoothing

        if smoothing_function is None and isinstance(smoothing_option_config, str):
            logger.warning(f"Invalid NLTK BLEU smoothing method '{smoothing_option_config}'. Defaulting to method1.")
            smoothing_function = SmoothingFunction().method1

        try:
            return corpus_bleu(
                self._collected_labels_tokens,
                self._collected_predictions_tokens,
                weights=weights,
                smoothing_function=smoothing_function
            )
        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0


class ROUGEScoreMetric(BaseMetric):
    """ Computes ROUGE score using the `rouge` library. """

    def __init__(self):
        """Initializes the ROUGEScoreMetric instance."""
        super().__init__()
        self._collected_predictions: List[str] = []
        self._collected_labels: List[str] = []
        self._rouge_calculator = Rouge()

    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self._collected_predictions = []
        self._collected_labels = []

    def update_state(self, predictions: List[str], labels: List[str]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        # Ensure inputs are lists of strings
        self._collected_predictions.extend([str(p) for p in predictions])
        self._collected_labels.extend([str(l) for l in labels])

    def _get_rouge_lib_key(self, r_metric_config_key: str) -> str:
        """Maps configuration key to `rouge` library key."""
        key_standardized = r_metric_config_key.lower().replace(" ", "") # Standardizza input
        
        if key_standardized == "rougel": return "rouge-l"
        if key_standardized == "rougelsum": return "rouge-lsum"
        
        # Check if already in library format e.g. "rouge-1", "rouge-l"
        if key_standardized.startswith("rouge-") and len(key_standardized.split('-')[-1]) > 0:
            return key_standardized
            
        # For formats like 'rouge1', 'rouge2' from config
        if key_standardized.startswith("rouge") and len(key_standardized) > 5 and key_standardized[5:].isalnum():
            return f"rouge-{key_standardized[5:]}"

        logger.warning(f"Unrecognized ROUGE metric config key '{r_metric_config_key}'. Using it as is, may not match library keys.")
        return r_metric_config_key

    def result(self) -> Dict[str, float]:
        """ Computes and returns the ROUGE score. """
        default_metrics_config = self._options.get("metrics", ['rouge-l'])
        default_stats_config = self._options.get("stats", ['f'])
        default_empty_or_error_result = {}
        for m_conf in default_metrics_config:
            for s_conf in default_stats_config:
                default_empty_or_error_result[f"{m_conf}_{s_conf}"] = 0.0
        if not default_empty_or_error_result:
            default_empty_or_error_result = {"rouge-l_f": 0.0}

        if not self._collected_predictions or not self._collected_labels:
            logger.warning("ROUGE: No predictions or labels collected.")
            return default_empty_or_error_result

        valid_hyps = []
        valid_refs = []

        for pred, ref in zip(self._collected_predictions, self._collected_labels):
            str_pred = str(pred).strip()
            str_ref = str(ref).strip()
            
            if str_pred and str_ref:
                valid_hyps.append(str_pred)
                valid_refs.append(str_ref)
            elif not str_pred and str_ref:
                logger.debug(f"ROUGE: Empty hypothesis for non-empty reference. This pair will effectively score 0 for ROUGE. Ref: '{str_ref[:50]}...'")

        if not valid_hyps:
            logger.warning("ROUGE: No valid (non-empty pairs of) predictions/references found after filtering. Returning 0.0 for scores.")
            return default_empty_or_error_result
        
        metrics_to_compute = self._options.get("metrics", ['rouge-l'])
        stats_to_return = self._options.get("stats", ['f'])
        
        try:
            scores_list_per_sentence = self._rouge_calculator.get_scores(valid_hyps, valid_refs, avg=False)
            
            num_valid_samples = len(scores_list_per_sentence)
            if num_valid_samples == 0:
                 return default_empty_or_error_result

            aggregated_scores: Dict[str, Dict[str, float]] = {}
            if scores_list_per_sentence:
                sample_score_item = scores_list_per_sentence[0]
                for rouge_type_key_from_lib in sample_score_item:
                    aggregated_scores[rouge_type_key_from_lib] = {stat_key: 0.0 for stat_key in sample_score_item[rouge_type_key_from_lib]}

            for score_item_dict in scores_list_per_sentence:
                for rouge_type_key_from_lib, stat_values_dict in score_item_dict.items():
                    if rouge_type_key_from_lib in aggregated_scores:
                        for stat_key, value in stat_values_dict.items():
                            if stat_key in aggregated_scores[rouge_type_key_from_lib]:
                                aggregated_scores[rouge_type_key_from_lib][stat_key] += value
                            else:
                                logger.warning(f"ROUGE: Unexpected stat '{stat_key}' from library for {rouge_type_key_from_lib}.")
                    else:
                         logger.warning(f"ROUGE: Unexpected rouge type '{rouge_type_key_from_lib}' from library.")
            
            averaged_scores_final: Dict[str, Dict[str, float]] = {}
            for rouge_type_key_from_lib, total_stats_dict in aggregated_scores.items():
                averaged_scores_final[rouge_type_key_from_lib] = {
                    stat_key: value / num_valid_samples for stat_key, value in total_stats_dict.items()
                }

            final_results_to_report = {}
            for r_metric_config_key in metrics_to_compute:
                lib_metric_key = self._get_rouge_lib_key(r_metric_config_key) 
                
                if lib_metric_key in averaged_scores_final:
                    for stat_config_key in stats_to_return:
                        if stat_config_key in averaged_scores_final[lib_metric_key]:
                            final_results_to_report[f"{r_metric_config_key}_{stat_config_key}"] = averaged_scores_final[lib_metric_key][stat_config_key]
                        else:
                            logger.warning(f"ROUGE: Configured stat '{stat_config_key}' not found for metric '{lib_metric_key}'. Available stats: {averaged_scores_final[lib_metric_key].keys()}")
                            final_results_to_report[f"{r_metric_config_key}_{stat_config_key}"] = 0.0
                elif r_metric_config_key.lower() == 'rougelsum' and 'rouge-l' in averaged_scores_final:
                    self.logger.debug("ROUGE: Treating 'rougeLsum' as 'rouge-l'.")
                    for stat_config_key in stats_to_return:
                        if stat_config_key in averaged_scores_final['rouge-l']:
                             final_results_to_report[f"rougeLsum_{stat_config_key}"] = averaged_scores_final['rouge-l'][stat_config_key]
                        else:
                             final_results_to_report[f"rougeLsum_{stat_config_key}"] = 0.0
                else:
                    logger.warning(f"ROUGE: Metric key '{lib_metric_key}' (from config key '{r_metric_config_key}') not found in calculated averaged scores. Available ROUGE types: {averaged_scores_final.keys()}")
                    for stat_config_key in stats_to_return:
                        final_results_to_report[f"{r_metric_config_key}_{stat_config_key}"] = 0.0
            
            return final_results_to_report if final_results_to_report else default_empty_or_error_result

        except Exception as e:
            logger.error(f"Error computing ROUGE score: {e}", exc_info=True)
            return {"rouge_exception": 1.0}


class PerplexityMetric(BaseMetric):
    """ Computes perplexity based on log probabilities of predictions. """

    def __init__(self):
        """Initializes the PerplexityMetric instance."""
        super().__init__()
        self.total_log_probs: float = 0.0
        self.total_elements: int = 0

    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self.total_log_probs = 0.0
        self.total_elements = 0

    def update_state(self, predictions: List[float], labels: List[Any]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch (log probabilities).
        :param labels: List of ground truth labels for the current batch (not used in this metric).
        """
        # Assuming predictions are per-token or per-sequence log probabilities
        batch_log_probs = np.array(predictions, dtype=np.float64)
        self.total_log_probs += np.sum(batch_log_probs)
        self.total_elements += len(batch_log_probs) # Or sum of token counts if predictions are per-token

    def result(self) -> float:
        """ Computes and returns the perplexity value. """
        if self.total_elements == 0:
            logger.warning("Perplexity: No elements to compute perplexity.")
            return float('inf') # Or 0.0, or a very high number. Inf is standard for PPL with no data.
        
        mean_log_probs = self.total_log_probs / self.total_elements
        cross_entropy = -mean_log_probs
        perplexity = np.exp(cross_entropy)
        
        if np.isnan(perplexity) or np.isinf(perplexity):
            logger.warning(f"Perplexity resulted in NaN or Inf (cross_entropy: {cross_entropy}).")
            return float('inf') # Standard representation for undefined PPL
        return perplexity


class ListAccumulatingMetric(BaseMetric):
    """Base class for metrics that accumulate lists of predictions and labels."""

    def __init__(self):
        """Initializes the ListAccumulatingMetric instance."""
        super().__init__()
        self._collected_predictions: List[Any] = []
        self._collected_labels: List[Any] = []

    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self._collected_predictions = []
        self._collected_labels = []

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        self._collected_predictions.extend(predictions)
        self._collected_labels.extend(labels)

    @abstractmethod
    def result(self) -> Union[float, Dict[str, float]]:
        pass


class ItemScoringAverageMetric(BaseMetric):
    """Base class for metrics that compute a score for each item and then average these scores."""

    def __init__(self):
        """Initializes the ItemScoringAverageMetric instance."""
        super().__init__()
        self.scores_sum: float = 0.0
        self.items_count: int = 0

    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self.scores_sum = 0.0
        self.items_count = 0

    @abstractmethod
    def _calculate_item_score(self, prediction: Any, label: Any) -> Optional[float]:
        """Subclasses must implement this to return a score for a single pred/label pair, or None to skip."""
        raise NotImplementedError

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        for pred_item, label_item in zip(predictions, labels):
            try:
                score = self._calculate_item_score(pred_item, label_item)
                if score is not None:
                    self.scores_sum += score
                    self.items_count += 1
            except Exception as e:
                logger.error(f"Error calculating item score in {self.__class__.__name__} for pred '{str(pred_item)[:50]}': {e}", exc_info=False)

    def result(self) -> float:
        """ Computes and returns the average score. """
        if self.items_count == 0:
            return 0.0
        return self.scores_sum / self.items_count


class METEORScoreMetric(ListAccumulatingMetric):
    """ Computes METEOR score using NLTK's meteor_score function. """

    def __init__(self):
        super().__init__()
        # Flag to ensure NLTK resources are checked/downloaded only once per instance
        self._nltk_resources_ensured: bool = False

    def _ensure_nltk_resources(self) -> bool:
        """
        Checks for necessary NLTK resources and downloads them if missing.
        Returns True if all resources are available or successfully downloaded, False otherwise.
        """
        if self._nltk_resources_ensured:
            return True

        required_resources = [
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'), # Open Multilingual Wordnet
            ('tokenizers/punkt', 'punkt')    # For tokenization by METEOR
        ]
        
        all_resources_ready = True
        for resource_path, resource_name in required_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info(f"NLTK resource '{resource_name}' for METEOR not found. Attempting download...")
                try:
                    nltk.download(resource_name, quiet=True)
                    logger.info(f"NLTK resource '{resource_name}' downloaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to download NLTK resource '{resource_name}': {e}. METEOR may be inaccurate.")
                    all_resources_ready = False # Mark as failure for this resource
            except Exception as e: # Catch any other errors during nltk.data.find
                logger.error(f"Unexpected error checking for NLTK resource '{resource_name}': {e}. METEOR may be inaccurate.")
                all_resources_ready = False
        
        if all_resources_ready:
            self._nltk_resources_ensured = True
        return all_resources_ready

    def result(self) -> float:
        """ Computes and returns the average METEOR score. """
        if not self._ensure_nltk_resources():
            logger.warning("METEOR score calculation skipped due to missing NLTK resources.")
            return 0.0

        if not self._collected_predictions or not self._collected_labels:
            logger.warning("METEOR: No predictions or labels collected to compute score.")
            return 0.0
        
        # Ensure all predictions and labels are strings before splitting
        try:
            tokenized_predictions = [str(p).split() for p in self._collected_predictions]
            # METEOR expects references as a list of lists of tokens (even for a single reference per hypothesis)
            tokenized_labels_meteor = [[str(l).split()] for l in self._collected_labels]
        except Exception as e:
            logger.error(f"METEOR: Error during tokenization of predictions/labels: {e}")
            return 0.0

        alpha = self._options.get("alpha", 0.9)  # Default NLTK METEOR parameters
        beta = self._options.get("beta", 3.0)
        gamma = self._options.get("gamma", 0.5)
        
        if not tokenized_predictions: # Should be caught by earlier check, but defensive
            return 0.0
        
        scores = []
        for hyp_tokens, refs_list_for_hyp in zip(tokenized_predictions, tokenized_labels_meteor):
            if not hyp_tokens and not any(refs_list_for_hyp): # Both empty or only ref list is empty of tokens
                 scores.append(0.0) # Define behavior for empty strings: score 0
                 continue
            if not hyp_tokens and any(ref_tok_list for ref_tok_list in refs_list_for_hyp if ref_tok_list): # Hyp empty, ref not
                 scores.append(0.0) # Hyp empty, ref not empty -> 0 score
                 continue

            try:
                # nltk_meteor_score expects references first, then hypothesis
                score = nltk_meteor_score(references=refs_list_for_hyp, 
                                          hypothesis=hyp_tokens,
                                          alpha=alpha, beta=beta, gamma=gamma)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error calculating METEOR for one sentence pair (hyp: '{' '.join(hyp_tokens)[:30]}...'): {e}")
                # Optionally append a 0 or skip, depending on desired error handling for individual pairs
        
        return np.mean(scores).item() if scores else 0.0


class JaccardSimilarityMetric(ItemScoringAverageMetric):
    """ Computes Jaccard similarity score between predictions and labels. """

    def _get_ngrams(self, text: str, n: int, normalize_text: bool):
        """
        Generates n-grams from the input text.
        :param text: Input text to process.
        :param n: Size of the n-grams.
        :param normalize_text: Whether to normalize the text (e.g., lowercasing).
        :return: Set of n-grams.
        """
        processed_text = str(text)
        if normalize_text:
            processed_text = processed_text.lower()
        words = processed_text.split()
        if len(words) < n:
            return set()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    def _calculate_item_score(self, prediction: Any, label: Any) -> Optional[float]:
        """
        Calculates the Jaccard similarity score for a single prediction-label pair.
        :param prediction: Model prediction.
        :param label: Ground truth label.
        :return: Jaccard similarity score.
        """
        ngram_n = self._options.get("ngram", 1)
        normalize = self._options.get("normalize", False)

        pred_ngrams = self._get_ngrams(str(prediction), ngram_n, normalize)
        label_ngrams = self._get_ngrams(str(label), ngram_n, normalize)
        
        intersection = len(pred_ngrams.intersection(label_ngrams))
        union = len(pred_ngrams.union(label_ngrams))
        
        return intersection / union if union > 0 else 0.0


class SemanticSimilarityMetric(ItemScoringAverageMetric):
    """ Computes semantic similarity score using SentenceTransformer model. """

    def __init__(self):
        """Initializes the SemanticSimilarityMetric instance."""
        super().__init__()
        self._model = None # SentenceTransformer model, initialized lazily
        self._model_name: Optional[str] = None 

    def _initialize_model(self):
        """Initializes the SentenceTransformer model."""
        # Initialize only if model name changed or not initialized
        model_name_option = self._options.get("model", 'all-MiniLM-L6-v2')
        if self._model is None or self._model_name != model_name_option:
            self._model_name = model_name_option
            try:
                from sentence_transformers import SentenceTransformer
                self.logger.info(f"Initializing SentenceTransformer model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                logger.error("sentence_transformers library not found. SemanticSimilarityMetric will not work.")
                self._model = None # Ensure model is None if import fails
                raise # Re-raise to signal failure clearly
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{self._model_name}': {e}")
                self._model = None
                raise # Re-raise

    def _calculate_item_score(self, prediction: Any, label: Any) -> Optional[float]:
        """
        Calculates the semantic similarity score for a single prediction-label pair.
        :param prediction: Model prediction.
        :param label: Ground truth label.
        :return: Semantic similarity score.
        """
        if self._model is None: # Attempt initialization if not done (e.g. first call)
            try:
                self._initialize_model()
                if self._model is None: # Still None after attempt
                    return None 
            except Exception: # Initialization failed
                 return None

        from sklearn.metrics.pairwise import cosine_similarity # Keep import local to method

        metric_type = self._options.get("metric_type", "cosine") # e.g. "cosine", "euclidean", "dot"
                                                                # Renamed from "metric" to avoid conflict with base class usage of self._options
        pred_emb = self._model.encode(str(prediction))
        label_emb = self._model.encode(str(label))
        
        score = 0.0
        if metric_type == 'cosine':
            score = cosine_similarity([pred_emb], [label_emb])[0][0]
        elif metric_type == 'euclidean':
            dist = np.linalg.norm(pred_emb - label_emb)
            score = 1 / (1 + dist) if dist is not None else 0.0 # Bounded score [0,1]
        elif metric_type == 'dot':
            score = np.dot(pred_emb, label_emb) # Unbounded, might need normalization depending on use case
        else:
            logger.warning(f"Unknown semantic similarity metric_type: {metric_type}. Returning 0 for this item.")
            return 0.0 # Or None to skip
        return float(score)


class DistinctNGramMetric(ListAccumulatingMetric):
    """ Computes distinct n-grams from the predictions. """

    def update_state(self, predictions: List[str], labels: List[Any]) -> None: # Labels are ignored
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch (not used).
        """
        self._collected_predictions.extend([str(p) for p in predictions])

    def result(self) -> Dict[str, float]:
        """ Computes and returns distinct n-gram scores. """
        if not self._collected_predictions:
            default_ngrams = self._options.get("ngrams", [1, 2])
            return {f"distinct_{n}": 0.0 for n in default_ngrams}

        texts = self._collected_predictions
        ngram_sizes = self._options.get("ngrams", [1, 2]) # e.g., [1, 2] for distinct-1, distinct-2
        results = {}

        for n_val in ngram_sizes:
            all_ngrams_in_corpus: List[str] = []
            if not texts: continue # Should be caught by initial check but defensive

            for text_item in texts:
                words = text_item.split()
                if len(words) < n_val: continue
                for i in range(len(words) - n_val + 1):
                    all_ngrams_in_corpus.append(" ".join(words[i:i+n_val]))
            
            if not all_ngrams_in_corpus:
                results[f"distinct_{n_val}"] = 0.0
                continue

            unique_ngrams_count = len(set(all_ngrams_in_corpus))
            total_ngrams_count = len(all_ngrams_in_corpus)
            results[f"distinct_{n_val}"] = unique_ngrams_count / total_ngrams_count if total_ngrams_count > 0 else 0.0
            
        return results


class WordEntropyMetric(ListAccumulatingMetric):
    """ Computes the entropy of words in the predictions. """

    def update_state(self, predictions: List[str], labels: List[Any]) -> None: # Labels are ignored
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch (not used).
        """
        self._collected_predictions.extend([str(p) for p in predictions])

    def result(self) -> float:
        """ Computes and returns the word entropy. """
        if not self._collected_predictions:
            return 0.0

        word_counts: Counter = Counter()
        total_words: int = 0
        
        for text_item in self._collected_predictions:
            words = text_item.split()
            word_counts.update(words)
            total_words += len(words)
            
        if total_words == 0:
            return 0.0
            
        probabilities = [count / total_words for count in word_counts.values() if count > 0]
        if not probabilities: # Handles case where all words were filtered or no words present
            return 0.0
            
        return -sum(p * math.log2(p) for p in probabilities) # Log base 2 for bits


class CorrelationMetric(ListAccumulatingMetric):
    """ Computes correlation between predictions and labels using scipy's pearsonr or spearmanr. """

    def _compute_correlation(self, func) -> float:
        """
        Computes the correlation using the provided function (pearsonr or spearmanr).
        :param func: The correlation function to use.
        :return: Computed correlation value.
        """
        if not self._collected_predictions or not self._collected_labels:
            return 0.0
        if len(self._collected_predictions) < 2 or len(self._collected_labels) < 2: # Correlation needs at least 2 points
            logger.warning(f"{self.__class__.__name__}: Need at least 2 data points to compute correlation. Got {len(self._collected_predictions)}.")
            return 0.0
        
        try:
            numeric_preds = [float(p) for p in self._collected_predictions]
            numeric_labels = [float(l) for l in self._collected_labels]
        except (ValueError, TypeError) as e:
            logger.error(f"{self.__class__.__name__}: Predictions or labels are not all numeric. Error: {e}")
            return 0.0
        
        try:
            correlation_result = func(numeric_preds, numeric_labels)
            corr_value = correlation_result[0] if isinstance(correlation_result, tuple) else correlation_result
            return corr_value if not np.isnan(corr_value) else 0.0
        except Exception as e:
            logger.error(f"Error computing correlation for {self.__class__.__name__}: {e}")
            return 0.0

class PearsonCorrelationMetric(CorrelationMetric):
    """ Computes Pearson correlation coefficient. """

    def result(self) -> float:
        return self._compute_correlation(pearsonr)

class SpearmanCorrelationMetric(CorrelationMetric):
    """ Computes Spearman rank-order correlation coefficient. """

    def result(self) -> float:
        return self._compute_correlation(spearmanr)


class SequenceLabelingMetrics(ListAccumulatingMetric):
    """ Computes sequence labeling metrics using seqeval library. """

    def result(self) -> Dict[str, float]:
        """
        Computes and returns sequence labeling metrics.
        :return: Dictionary with precision, recall, and F1 scores.
        """
        if not self._collected_predictions or not self._collected_labels:
            return {"overall_precision": 0.0, "overall_recall": 0.0, "overall_f1": 0.0}

        scheme = self._options.get("scheme", None) 
        mode = self._options.get("mode", "default") # 'strict' or 'default'
        average_type = self._options.get("average_type", "micro avg") # 'micro avg', 'macro avg', 'weighted avg'
        
        try:
            report = seqeval_classification_report(
                y_true=self._collected_labels, 
                y_pred=self._collected_predictions, 
                mode=mode, 
                scheme=scheme, 
                output_dict=True,
                zero_division=0 
            )
            
            if average_type in report:
                avg_metrics = report[average_type]
                return {
                    f'overall_precision': avg_metrics.get('precision', 0.0),
                    f'overall_recall': avg_metrics.get('recall', 0.0),
                    f'overall_f1': avg_metrics.get('f1-score', 0.0) # f1-score is the key from seqeval
                }
            else:
                logger.warning(f"Average type '{average_type}' not found in seqeval report. Available keys: {list(report.keys())}. Defaulting overall scores to 0.")
                # Fallback to first available class if no standard average is found, or just return 0s
                first_class_key = next((k for k in report if isinstance(report[k], dict) and 'f1-score' in report[k]), None)
                if first_class_key and isinstance(report[first_class_key], dict):
                     class_metrics = report[first_class_key]
                     return {
                        f'{first_class_key}_precision': class_metrics.get('precision',0.0),
                        f'{first_class_key}_recall': class_metrics.get('recall',0.0),
                        f'{first_class_key}_f1': class_metrics.get('f1-score',0.0),
                     }
                return {"overall_precision": 0.0, "overall_recall": 0.0, "overall_f1": 0.0, "error": f"Average type '{average_type}' not found."}

        except Exception as e:
            logger.error(f"Error computing SequenceLabelingMetrics with seqeval: {e}", exc_info=True)
            return {"overall_precision": 0.0, "overall_recall": 0.0, "overall_f1": 0.0, "error": str(e)}


class BERTScoreMetric(ListAccumulatingMetric):
    """ Computes BERTScore using the `bert_score` library. """

    def result(self) -> Dict[str, float]:
        """
        Computes and returns BERTScore.
        :return: Dictionary with precision, recall, and F1 scores.
        """
        if not self._collected_predictions or not self._collected_labels:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

        lang = self._options.get("lang", "en")
        model_type = self._options.get("model_type", None) 
        
        bertscore_fn_options = {
            "lang": lang,
            "verbose": self._options.get("verbose", False),
            "idf": self._options.get("idf", False), # Pass idf from options
            "batch_size": self._options.get("bertscore_batch_size", 64), # Specific option for bert_score's internal batching
            "device": self._options.get("device", None) # Allow specifying device for bert_score
        }
        if model_type:
            bertscore_fn_options["model_type"] = model_type
        
        try:
            # Ensure all predictions and labels are strings
            str_predictions = [str(p) for p in self._collected_predictions]
            str_labels = [str(l) for l in self._collected_labels]

            precision, recall, f1 = bert_score.score(
                cands=str_predictions,
                refs=str_labels,
                **bertscore_fn_options
            )
            return {
                "bertscore_precision": precision.mean().item(),
                "bertscore_recall": recall.mean().item(),
                "bertscore_f1": f1.mean().item()
            }
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}", exc_info=True)
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0, "error": str(e)}


class HFcommonPipelineMetric(BaseMetric):
    """Base class for metrics using Hugging Face pipelines."""

    def __init__(self):
        """Initializes the HFcommonPipelineMetric instance."""
        super().__init__()
        self.scores_sum: float = 0.0
        self.items_count: int = 0 # Renamed from self.count for clarity
        self._pipeline = None
        self._pipeline_task_name: str = "" 
        self._pipeline_model_name_default: str = "" # Default model for the pipeline if not in options
        self._pipeline_model_name_actual: Optional[str] = None # Actual model name used after checking options
        self._score_key_in_result: str = "score" 
        self._result_if_no_data: Any = 0.0 # What to return if no items were processed


    def _initialize_pipeline(self) -> bool:
        """
        Initializes the Hugging Face pipeline if not already initialized or if model name has changed.
        :return: True if pipeline is successfully initialized, False otherwise.
        """
        # Initialize only if model name from options has changed or not initialized
        model_name_from_options = self._options.get("model", self._pipeline_model_name_default)
        
        if self._pipeline is not None and self._pipeline_model_name_actual == model_name_from_options:
            return True # Already initialized with the correct model

        self._pipeline_model_name_actual = model_name_from_options
        if not self._pipeline_model_name_actual: # Check if a model name is actually determined
            logger.error(f"No model name specified or defaulted for {self.__class__.__name__} pipeline.")
            self._pipeline = None # Ensure pipeline is None
            return False
        
        try:
            # Specify device, can be made configurable via self._options.get("device", "cpu") for example
            # Using -1 for CPU to avoid assuming CUDA for these helper pipelines by default.
            # Or self._options.get("pipeline_device")
            pipeline_device = self._options.get("device", -1 if not torch.cuda.is_available() else 0) 
            logger.info(f"Initializing HF pipeline '{self._pipeline_task_name}' with model '{self._pipeline_model_name_actual}' on device '{pipeline_device}' for {self.__class__.__name__}")
            self._pipeline = pipeline(self._pipeline_task_name, model=self._pipeline_model_name_actual, device=pipeline_device)
            return True
        except Exception as e:
            logger.error(f"Failed to load HF pipeline for {self.__class__.__name__} (task: {self._pipeline_task_name}, model: {self._pipeline_model_name_actual}): {e}", exc_info=True)
            self._pipeline = None
            return False


    def reset_state(self) -> None:
        """Resets the internal state of the metric."""
        self.scores_sum = 0.0
        self.items_count = 0
        # Do not reset self._pipeline here to allow reuse across evaluations if config doesn't change.
        # It will be re-initialized if model_name_from_options changes.

    @abstractmethod
    def _process_pipeline_output(self, output: Any, prediction: Any, label: Any) -> Optional[float]:
        """Subclasses must implement this to extract a meaningful score from the pipeline's output."""
        raise NotImplementedError

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Updates the metric's state with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        if not self._initialize_pipeline() or self._pipeline is None:
            logger.warning(f"Pipeline for {self.__class__.__name__} not available or failed to initialize. Skipping update.")
            return

        for pred_item, label_item in zip(predictions, labels):
            try:
                # Subclasses might need to format input to the pipeline differently
                pipeline_input = self._prepare_pipeline_input(pred_item, label_item)
                pipeline_output = self._pipeline(pipeline_input) if isinstance(pipeline_input, (str, list)) else self._pipeline(**pipeline_input)
                
                score = self._process_pipeline_output(pipeline_output, pred_item, label_item)
                if score is not None:
                    self.scores_sum += score
                    self.items_count += 1
            except Exception as e:
                logger.error(f"Error processing item in {self.__class__.__name__} for pred '{str(pred_item)[:50]}': {e}", exc_info=False) # exc_info=False for less verbose logs by default

    def _prepare_pipeline_input(self, prediction: Any, label: Any) -> Any:
        """
        Prepares input for the Hugging Face pipeline.
        Default implementation assumes prediction is the primary text.
        Subclasses should override if different input structure is needed.
        :param prediction: Model prediction.
        :param label: Ground truth label (not used in default).
        :return: Input for the pipeline.
        """
        return str(prediction) # Default: pipeline takes the prediction text

    def result(self) -> Any:
        """ Computes and returns the average score. """
        if self.items_count == 0:
            return self._result_if_no_data
        return self.scores_sum / self.items_count


class FactualConsistencyMetric(HFcommonPipelineMetric):
    """ Computes factual consistency using a question-answering pipeline. """

    def __init__(self):
        """Initializes the FactualConsistencyMetric instance."""
         # Initialize the base class
         # Set default model and task name for the pipeline
         # Set the key in the result dict to extract the score
        super().__init__()
        self._pipeline_task_name = "question-answering"
        self._pipeline_model_name_default = "deepset/tinyroberta-squad2" 
        self._score_key_in_result = "score" 

    def _prepare_pipeline_input(self, prediction: Any, label: Any) -> Dict[str, str]:
        """
        Prepares input for the Hugging Face pipeline.
        :param prediction: Model prediction (question).
        :param label: Ground truth label (context).
        :return: Dictionary with 'question' and 'context' keys.
        """
        # Prediction is the generated answer, Label is the context document
        return {"question": str(prediction), "context": str(label)}

    def _process_pipeline_output(self, output: Any, prediction: Any, label: Any) -> Optional[float]:
        """
        Processes the output from the pipeline to extract the score.
        :param output: Output from the pipeline.
        :param prediction: Model prediction (question).
        :param label: Ground truth label (context).
        :return: Confidence score for the factual consistency.
        """
        # QA pipeline output is a dict e.g. {'score': 0.99, 'start': 3, 'end': 10, 'answer': 'extracted'}
        if isinstance(output, dict) and self._score_key_in_result in output:
            confidence_score = output[self._score_key_in_result]
            
            # Option to use a threshold for binary consistency
            threshold = self._options.get("consistency_threshold", None)
            if threshold is not None:
                return 1.0 if confidence_score >= threshold else 0.0
            return float(confidence_score) # Return raw confidence
        else:
            logger.warning(f"FactualConsistency: Unexpected pipeline output format: {output}")
            return None


class ToxicityScoreMetric(HFcommonPipelineMetric):
    """ Computes toxicity score using a text-classification pipeline. """

    def __init__(self):
        """Initializes the ToxicityScoreMetric instance."""
        super().__init__()
        self._pipeline_task_name = "text-classification"
        self._pipeline_model_name_default = "unitary/toxic-bert"
        self._target_label_for_score: str = "TOXICITY" # Default label to extract score for
        self._result_if_no_data = {self._target_label_for_score.lower(): 0.0} 

    def _initialize_pipeline(self) -> bool:
        """
        Initializes the Hugging Face pipeline for toxicity scoring.
        :return: True if pipeline is successfully initialized, False otherwise.
        """
        # Update target label from options before initializing (in case model output labels change)
        self._target_label_for_score = str(self._options.get("target_label", "TOXICITY")).upper()
        self._result_if_no_data = {self._target_label_for_score.lower(): 0.0}
        return super()._initialize_pipeline()


    def _prepare_pipeline_input(self, prediction: Any, label: Any) -> Any: # Label is ignored for toxicity
        """
        Prepares input for the Hugging Face pipeline.
        :param prediction: Model prediction (text to classify).
        :param label: Ground truth label (not used).
        :return: Input for the pipeline.
        """
        return str(prediction)

    def _process_pipeline_output(self, output: Any, prediction: Any, label: Any) -> Optional[float]:
        """
        Processes the output from the pipeline to extract the toxicity score.
        :param output: Output from the pipeline.
        :param prediction: Model prediction (text).
        :param label: Ground truth label (not used).
        :return: Toxicity score for the prediction.
        """
        # Text-classification pipeline can return a list of dicts (if top_k=None or >1)
        # or a single dict (if top_k=1, which is often default for single label tasks)
        # [{'label': 'LABEL_1', 'score': 0.9}, {'label': 'LABEL_0', 'score': 0.1}]
        if not isinstance(output, list): # Ensure it's a list for uniform processing
            output = [output]

        for res_item in output:
            if isinstance(res_item, dict) and 'label' in res_item and 'score' in res_item:
                if res_item['label'].upper() == self._target_label_for_score:
                    return float(res_item['score'])
            else:
                logger.warning(f"ToxicityScoreMetric: Unexpected item format in pipeline result: {res_item}")
        
        # If target label not found in any of the results
        # Log available labels for debugging if a list of dicts was returned
        available_labels = [item.get('label', 'N/A') for item in output if isinstance(item, dict)]
        logger.warning(f"ToxicityScoreMetric: Target label '{self._target_label_for_score}' not found in pipeline results. Available: {available_labels}")
        return None 

    def result(self) -> Dict[str, float]: # Override to always return a dict for the target label
        """
        Computes and returns the average toxicity score.
        :return: Dictionary with the average toxicity score.
        """
        avg_score = super().result() # This gets the float average from base class
        return {self._target_label_for_score.lower(): float(avg_score) if isinstance(avg_score, (float, int)) else 0.0}