from abc import abstractmethod
from typing import Any, List, Union, Dict
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge import Rouge
import bert_score
from collections import Counter
import math
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report as seqeval_classification_report
from transformers import pipeline
TRANSFORMERS_AVAILABLE = True
from .base_metrics import BaseMetric
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AccuracyMetric(BaseMetric):
    """Class for computing accuracy."""
    def __init__(self):
        super().__init__()
        self.correct_predictions = 0
        self.total_predictions = 0

    def reset_state(self) -> None:
        """Reset the internal state of the metric."""
        self.correct_predictions = 0
        self.total_predictions = 0

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Update the metric with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        # top_k = self._options.get("top_k", None) # Example if options were needed
        # if top_k:
        #     logger.warning("Top-k accuracy not implemented in this stateful version yet.")
        #     # Implement top-k logic here if needed
        
        for pred, label in zip(predictions, labels):
            if pred == label: # Simple equality check, assuming processed inputs
                self.correct_predictions += 1
            self.total_predictions += 1

    def result(self) -> float:
        """Compute the final accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


class SklearnScoreMetric(BaseMetric): # Base for Precision, Recall, F1
    """Base class for sklearn metrics that collect all preds/labels."""
    def __init__(self):
        super().__init__()
        self._collected_predictions: List[Any] = []
        self._collected_labels: List[Any] = []

    def reset_state(self) -> None:
        """Reset the internal state of the metric."""
        self._collected_predictions = []
        self._collected_labels = []

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Update the metric with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        self._collected_predictions.extend(predictions)
        self._collected_labels.extend(labels)

    def _compute_score(self, score_func, **kwargs) -> float:
        """
        Compute the score using the provided sklearn function.
        :param score_func: The sklearn scoring function to use (e.g., precision_score).
        :param kwargs: Additional keyword arguments for the scoring function.
        :return: Computed score.
        """
        if not self._collected_predictions or not self._collected_labels:
            logger.warning(f"{self.__class__.__name__}: No data collected to compute score.")
            return 0.0
        # Ensure labels are not empty and lengths match if sklearn func is sensitive
        if len(self._collected_predictions) != len(self._collected_labels):
             logger.error(f"{self.__class__.__name__}: Preds and labels length mismatch. Cannot compute score.")
             return 0.0 # Or raise error

        # Pass relevant options from self._options
        average = self._options.get("average", "weighted")
        zero_division = self._options.get("zero_division", 0) # sklearn's default is "warn", use 0 or 1
        
        # Add other relevant sklearn options from self._options if needed
        sklearn_options = {"average": average, "zero_division": zero_division}
        # Filter out options not applicable to the specific score_func if necessary
        
        try:
            return score_func(self._collected_labels, self._collected_predictions, **sklearn_options)
        except Exception as e:
            logger.error(f"Error computing {self.__class__.__name__} with sklearn: {e}")
            return 0.0


class PrecisionMetric(SklearnScoreMetric):
    """Class for computing precision."""
    def result(self) -> float:
        return self._compute_score(precision_score)

class RecallMetric(SklearnScoreMetric):
    """Class for computing recall."""
    def result(self) -> float:
        return self._compute_score(recall_score)

class F1ScoreMetric(SklearnScoreMetric):
    """Class for computing F1 score."""
    def result(self) -> float:
        return self._compute_score(f1_score)


class ExactMatchMetric(BaseMetric):
    """Class for computing exact match with normalization options."""
    def __init__(self):
        super().__init__()
        self.match_count = 0
        self.total_count = 0

    def reset_state(self) -> None:
        """Reset the internal state of the metric."""
        self.match_count = 0
        self.total_count = 0

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Update the metric with a batch of predictions and labels
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        normalize = self._options.get("normalize", False)
        ignore_case = self._options.get("ignore_case", False)
        ignore_punct = self._options.get("ignore_punct", False)

        for pred_item, label_item in zip(predictions, labels):
            pred_str = str(pred_item)
            label_str = str(label_item)

            if normalize: # Apply normalization from options
                if ignore_case:
                    pred_str = pred_str.lower()
                    label_str = label_str.lower()
                if ignore_punct:
                    # This could be slow for very long strings. Consider re.sub if performance is an issue.
                    pred_str = ''.join(c for c in pred_str if c.isalnum() or c.isspace()) # Example: keep alphanumeric and space
                    label_str = ''.join(c for c in label_str if c.isalnum() or c.isspace()) # Be careful with this
                    # Original: pred_str = ''.join(c for c in pred_str if c not in '.,;!?\'"')

            if pred_str == label_str:
                self.match_count += 1
            self.total_count += 1

    def result(self) -> float:
        """Compute the final exact match score."""
        if self.total_count == 0:
            return 0.0
        return self.match_count / self.total_count


class BLEUScoreMetric(BaseMetric):
    """Class for computing BLEU score."""
    def __init__(self):
        super().__init__()
        self._collected_predictions_tokens: List[List[str]] = []
        self._collected_labels_tokens: List[List[List[str]]] = [] # List of (list of reference tokens)

    def reset_state(self) -> None:
        """Reset the internal state of the metric."""
        self._collected_predictions_tokens = []
        self._collected_labels_tokens = []

    def update_state(self, predictions: List[str], labels: List[Union[str, List[str]]]) -> None:
        """
        Update the metric with a batch of predictions and labels.
        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        # Predictions are expected as list of strings (sentences)
        # Labels can be list of strings (single ref per pred) or list of list of strings (multiple refs per pred)
        for pred_text, label_item in zip(predictions, labels):
            self._collected_predictions_tokens.append(pred_text.split()) # Tokenize by space
            if isinstance(label_item, str):
                self._collected_labels_tokens.append([label_item.split()])
            elif isinstance(label_item, list): # Assuming list of reference strings
                self._collected_labels_tokens.append([ref_text.split() for ref_text in label_item])
            else:
                logger.warning("Unsupported label format for BLEU, skipping item.")


    def result(self) -> float:
        if not self._collected_predictions_tokens or not self._collected_labels_tokens:
            return 0.0

        weights = self._options.get("weights", (0.25, 0.25, 0.25, 0.25)) # Default BLEU-4
        smoothing_option = self._options.get("smoothing", True) # bool or SmoothingFunction
        
        smoothing_function = None
        if isinstance(smoothing_option, bool) and smoothing_option:
            smoothing_function = SmoothingFunction().method1 # Default NLTK smoothing
        elif callable(smoothing_option):
            smoothing_function = smoothing_option
        
        try:
            # corpus_bleu expects list of list of references, and list of hypotheses
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
    """Class for computing ROUGE score."""
    def __init__(self):
        super().__init__()
        self._collected_predictions: List[str] = []
        self._collected_labels: List[str] = [] # Assuming single reference string per prediction for simplicity here
                                              # If multiple references, labels might be List[List[str]]
                                              # and rouge library needs to handle it or adapt.
        self._rouge_calculator = Rouge()


    def reset_state(self) -> None:
        self._collected_predictions = []
        self._collected_labels = []

    def update_state(self, predictions: List[str], labels: List[str]) -> None:
        self._collected_predictions.extend(predictions)
        self._collected_labels.extend(labels)

    def result(self) -> Dict[str, float]:
        if not self._collected_predictions or not self._collected_labels:
            return {"rouge-l-f": 0.0} # Default structure

        # Options for Rouge, e.g., specific metrics like ['rouge-1', 'rouge-2', 'rouge-l']
        # and stats like ['f', 'p', 'r']
        # These should be passed via set_options and accessed via self._options
        # Example default from original code:
        metrics_to_compute = self._options.get("metrics", ['rouge-l']) # e.g. ['rouge1', 'rouge2', 'rougeL']
        stats_to_return = self._options.get("stats", ['f'])       # e.g. ['f', 'p', 'r']
        
        try:
            scores = self._rouge_calculator.get_scores(self._collected_predictions, self._collected_labels, avg=True)
            
            final_results = {}
            for r_metric in metrics_to_compute: # e.g. 'rouge1', 'rouge2', 'rougeL'
                lib_metric_key = r_metric.replace('rougeLsum', 'rouge-lsum').replace('rouge', 'rouge-') # Basic mapping
                if lib_metric_key.endswith('-') and len(lib_metric_key) > 6: # e.g. rouge-1, rouge-2, rouge-L
                     if lib_metric_key[-2].isalpha() and lib_metric_key[-2].islower(): # rouge-l -> rouge-L
                         lib_metric_key = lib_metric_key[:-2] + lib_metric_key[-2].upper() + lib_metric_key[-1:]


                if lib_metric_key in scores:
                    for stat in stats_to_return:
                        if stat in scores[lib_metric_key]:
                            final_results[f"{r_metric}_{stat}"] = scores[lib_metric_key][stat] # Original code used f"{metric}-{stat}"
                        else:
                            logger.warning(f"Stat '{stat}' not found for ROUGE metric '{lib_metric_key}'.")
                else:
                    # Handle cases like 'rougeLsum' if the library provides it differently or it needs special calculation
                    if r_metric == 'rougeLsum' and 'rouge-l' in scores: # Often rougeLsum is just rougeL for abstractive
                        for stat in stats_to_return:
                             if stat in scores['rouge-l']:
                                final_results[f"rougeLsum_{stat}"] = scores['rouge-l'][stat]
                             else:
                                logger.warning(f"Stat '{stat}' not found for ROUGE metric 'rouge-l' (used for rougeLsum).")
                    else:
                        logger.warning(f"ROUGE metric '{r_metric}' (mapped to '{lib_metric_key}') not found in scores: {scores.keys()}")
            return final_results if final_results else {"rouge_error": 1.0}

        except Exception as e:
            logger.error(f"Error computing ROUGE score: {e}")
            return {"rouge_error": 1.0} # Default error structure

class PerplexityMetric(BaseMetric):
    """Class for computing perplexity."""
    def __init__(self):
        super().__init__()
        self.total_log_probs = 0.0
        self.total_elements = 0 # Could be tokens or sequences depending on how PPL is defined for the task

    def reset_state(self) -> None:
        self.total_log_probs = 0.0
        self.total_elements = 0

    def update_state(self, predictions: List[float], labels: List[Any]) -> None: 
        batch_log_probs = np.array(predictions, dtype=np.float64)
        self.total_log_probs += np.sum(batch_log_probs)
        self.total_elements += len(batch_log_probs)

    def result(self) -> float:
        if self.total_elements == 0:
            return 0.0 # Or a very high number for PPL, or NaN
        
        mean_log_probs = self.total_log_probs / self.total_elements
        cross_entropy = -mean_log_probs
        perplexity = np.exp(cross_entropy)
        
        if np.isnan(perplexity) or np.isinf(perplexity):
            logger.warning(f"Perplexity resulted in NaN or Inf (cross_entropy: {cross_entropy}). Returning large value.")
            return float(1e12) # Return a large number instead of NaN/Inf
        return perplexity


class ListAccumulatingMetric(BaseMetric): # Base for metrics that just collect and then compute
    """Base class for metrics that accumulate all predictions and labels then compute."""
    def __init__(self):
        super().__init__()
        self._collected_predictions: List[Any] = []
        self._collected_labels: List[Any] # For some metrics, labels might be contexts etc.

    def reset_state(self) -> None:
        self._collected_predictions = []
        self._collected_labels = [] # Ensure this is also initialized

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        self._collected_predictions.extend(predictions)
        self._collected_labels.extend(labels) # Assuming labels are also collected

    @abstractmethod # Subclasses must implement their specific result calculation
    def result(self) -> Union[float, Dict[str, float]]:
        pass


class METEORScoreMetric(ListAccumulatingMetric):
    """Class for computing METEOR score."""
    def result(self) -> float:
        from nltk.translate.meteor_score import meteor_score # Import here to avoid error if NLTK/meteor not fully set up

        if not self._collected_predictions or not self._collected_labels:
            return 0.0

        # METEOR expects tokenized strings. Predictions/labels should be strings.
        # The original code tokenized them if they weren't lists of tokens.
        # Assuming _collected_predictions and _collected_labels are lists of strings.
        
        tokenized_predictions = [p.split() for p in self._collected_predictions]
        # meteor_score expects references as list of lists of tokens (even for single ref)
        tokenized_labels_meteor = [[l.split()] for l in self._collected_labels]

        alpha = self._options.get("alpha", 0.9)
        beta = self._options.get("beta", 3.0)
        gamma = self._options.get("gamma", 0.5)
        
        if not tokenized_predictions: return 0.0
        
        scores = []
        for hyp_tokens, refs_tokens_list_for_one_hyp in zip(tokenized_predictions, tokenized_labels_meteor):
            # refs_tokens_list_for_one_hyp is like [['ref1', 'tok1'], ['ref2', 'tok2']] if multiple actual refs
            # For single reference, it's [['ref', 'tok']]
            try:
                score = meteor_score(references=refs_tokens_list_for_one_hyp, hypothesis=hyp_tokens,
                                     alpha=alpha, beta=beta, gamma=gamma)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error calculating METEOR for one sentence: {e}")
        
        return np.mean(scores) if scores else 0.0


class JaccardSimilarityMetric(BaseMetric):
    """Class for computing Jaccard similarity between texts."""
    def __init__(self):
        super().__init__()
        self.scores_sum = 0.0
        self.count = 0

    def reset_state(self) -> None:
        self.scores_sum = 0.0
        self.count = 0

    def update_state(self, predictions: List[str], labels: List[str]) -> None:
        ngram = self._options.get("ngram", 1)
        normalize_text = self._options.get("normalize", False) # Renamed from 'normalize' to avoid conflict

        def get_ngrams(text: str, n: int):
            if normalize_text: # Use the option
                text = text.lower()
            words = text.split()
            return set([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])

        for pred_text, label_text in zip(predictions, labels):
            pred_ngrams = get_ngrams(str(pred_text), ngram)
            label_ngrams = get_ngrams(str(label_text), ngram)
            
            intersection = len(pred_ngrams.intersection(label_ngrams))
            union = len(pred_ngrams.union(label_ngrams))
            
            score = intersection / union if union > 0 else 0.0
            self.scores_sum += score
            self.count += 1
            
    def result(self) -> float:
        if self.count == 0:
            return 0.0
        return self.scores_sum / self.count


class SemanticSimilarityMetric(BaseMetric):
    """Class for computing semantic similarity using sentence embeddings."""
    def __init__(self):
        super().__init__()
        self.scores_sum = 0.0
        self.count = 0
        self._model = None # SentenceTransformer model, initialized lazily

    def _initialize_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self._options.get("model", 'all-MiniLM-L6-v2')
                self._model = SentenceTransformer(model_name)
            except ImportError:
                logger.error("sentence_transformers library not found. SemanticSimilarityMetric will not work.")
                raise
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise


    def reset_state(self) -> None:
        self.scores_sum = 0.0
        self.count = 0
        # self._model = None # Don't reset model to avoid reloading, unless config changes

    def update_state(self, predictions: List[str], labels: List[str]) -> None:
        try:
            self._initialize_model() # Ensure model is loaded
            if self._model is None: return # Failed to load
        except Exception:
             return # Error already logged

        from sklearn.metrics.pairwise import cosine_similarity # Import here

        metric_type = self._options.get("metric", "cosine")

        # Encode in batches for efficiency if sentence_transformers supports it well for many short texts
        try:
            # It's often better to encode all predictions and labels in a batch at once
            all_texts_in_batch = predictions + labels
            if not all_texts_in_batch: return

            all_embeddings = self._model.encode(all_texts_in_batch)
            num_preds = len(predictions)
            pred_embeddings = all_embeddings[:num_preds]
            label_embeddings = all_embeddings[num_preds:]

            for i in range(num_preds):
                pred_emb = pred_embeddings[i]
                label_emb = label_embeddings[i]
                score = 0.0
                if metric_type == 'cosine':
                    score = cosine_similarity([pred_emb], [label_emb])[0][0]
                elif metric_type == 'euclidean':
                    dist = np.linalg.norm(pred_emb - label_emb)
                    score = 1 / (1 + dist) if dist is not None else 0.0
                elif metric_type == 'dot':
                    score = np.dot(pred_emb, label_emb)
                else:
                    logger.warning(f"Unknown semantic similarity metric type: {metric_type}")
                    continue
                
                self.scores_sum += score
                self.count += 1
        except Exception as e:
            logger.error(f"Error during semantic similarity batch update: {e}")


    def result(self) -> float:
        if self.count == 0:
            return 0.0
        return self.scores_sum / self.count


class DistinctNGramMetric(ListAccumulatingMetric): # Inherits reset_state, update_state
    """Computes distinct n-gram metrics for text diversity from collected texts."""
    # update_state will collect predictions (texts). Labels are ignored for this metric.

    def update_state(self, predictions: List[str], labels: List[Any]) -> None: # Labels ignored
        self._collected_predictions.extend(predictions) # predictions are texts

    def result(self) -> Dict[str, float]:
        if not self._collected_predictions:
            return {"distinct_1": 0.0, "distinct_2": 0.0} # Default if no texts

        texts = self._collected_predictions
        ngram_sizes = self._options.get("ngrams", [1, 2]) # e.g., [1, 2] for distinct-1, distinct-2
        results = {}

        for n in ngram_sizes:
            all_ngrams_in_corpus = []
            if not texts: continue

            for text in texts:
                if not isinstance(text, str):
                    logger.warning(f"DistinctNGramMetric: item is not a string: {type(text)}. Skipping.")
                    continue
                words = text.split()
                if len(words) < n : continue
                for i in range(len(words) - n + 1):
                    all_ngrams_in_corpus.append(" ".join(words[i:i+n]))
            
            if not all_ngrams_in_corpus:
                results[f"distinct_{n}"] = 0.0
                continue

            unique_ngrams = len(set(all_ngrams_in_corpus))
            total_ngrams = len(all_ngrams_in_corpus)
            results[f"distinct_{n}"] = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
            
        return results


class WordEntropyMetric(ListAccumulatingMetric):
    """Computes word-level entropy for text generation."""
    # update_state will collect predictions (texts). Labels are ignored.
    def update_state(self, predictions: List[str], labels: List[Any]) -> None: # Labels ignored
        self._collected_predictions.extend(predictions) # predictions are texts

    def result(self) -> float:
        if not self._collected_predictions:
            return 0.0

        texts = self._collected_predictions
        word_counts = Counter()
        total_words = 0
        
        for text in texts:
            if not isinstance(text, str):
                 logger.warning(f"WordEntropyMetric: item is not a string: {type(text)}. Skipping.")
                 continue
            words = text.split()
            word_counts.update(words)
            total_words += len(words)
            
        if total_words == 0:
            return 0.0
            
        probs = [count / total_words for count in word_counts.values() if count > 0]
        if not probs:
            return 0.0
            
        return -sum(p * math.log(p, 2) for p in probs) # Log base 2 for bits


class CorrelationMetric(ListAccumulatingMetric): # Base for Pearson, Spearman
    """Base for correlation metrics."""
    def _compute_correlation(self, func) -> float:
        if not self._collected_predictions or not self._collected_labels:
            return 0.0
        if len(self._collected_predictions) < 2 or len(self._collected_labels) < 2: # Correlation needs at least 2 points
            return 0.0
        # Ensure predictions and labels are numeric
        try:
            numeric_preds = [float(p) for p in self._collected_predictions]
            numeric_labels = [float(l) for l in self._collected_labels]
        except ValueError:
            logger.error(f"{self.__class__.__name__}: Predictions or labels are not numeric.")
            return 0.0
        
        try:
            corr_value, _ = func(numeric_preds, numeric_labels)
            return corr_value if not np.isnan(corr_value) else 0.0
        except Exception as e:
            logger.error(f"Error computing correlation for {self.__class__.__name__}: {e}")
            return 0.0

class PearsonCorrelationMetric(CorrelationMetric):
    def result(self) -> float:
        return self._compute_correlation(pearsonr)

class SpearmanCorrelationMetric(CorrelationMetric):
    def result(self) -> float:
        return self._compute_correlation(spearmanr)


class SequenceLabelingMetrics(ListAccumulatingMetric):
    """Computes sequence labeling metrics (NER, POS tagging, etc.) using seqeval."""
    # update_state collects predictions (List[List[str]]) and labels (List[List[str]])
    
    def result(self) -> Dict[str, float]:
        if not self._collected_predictions or not self._collected_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0} # Default structure

        # Options for seqeval
        scheme = self._options.get("scheme", None) # e.g., 'IOB2', 'BIOES'
        mode = self._options.get("mode", "default") # 'strict' or 'default'
        
        try:
            report = seqeval_classification_report(
                y_true=self._collected_labels, 
                y_pred=self._collected_predictions, 
                mode=mode, 
                scheme=scheme, 
                output_dict=True,
                zero_division=0 # Consistent with sklearn metrics
            )
            
            # Extract overall metrics (micro or weighted average)
            if 'micro avg' in report: # Often used for token-level overall performance
                return {
                    'precision': report['micro avg']['precision'],
                    'recall': report['micro avg']['recall'],
                    'f1': report['micro avg']['f1-score']
                }
            elif 'weighted avg' in report: # Fallback if micro avg not present
                return {
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1': report['weighted avg']['f1-score']
                }
            else: # No overall average found, try to report something or default
                logger.warning("No 'micro avg' or 'weighted avg' in seqeval report. Check report structure.")
                # You might want to average class F1s or return a specific class's score
                # For now, returning default on failure to find standard averages.
                first_class_key = next(iter(k for k in report.keys() if isinstance(report[k], dict) and 'f1-score' in report[k]), None)
                if first_class_key:
                     return {
                        'precision': report[first_class_key]['precision'],
                        'recall': report[first_class_key]['recall'],
                        'f1': report[first_class_key]['f1-score'],
                        'class_reported': first_class_key
                    }
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": "No standard average in report"}

        except Exception as e:
            logger.error(f"Error computing SequenceLabelingMetrics with seqeval: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)}


class BERTScoreMetric(ListAccumulatingMetric):
    """Computes BERTScore for text generation evaluation."""
    def result(self) -> Dict[str, float]:
        if not self._collected_predictions or not self._collected_labels:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

        # Options for bert_score
        lang = self._options.get("lang", "en")
        model_type = self._options.get("model_type", None) # bert_score lib will use default if None
        # Other bert_score.score params: num_layers, verbose, idf, batch_size, device, etc.
        # Can be passed via self._options.get('bertscore_options', {})
        
        bertscore_options = {
            "lang": lang,
            "verbose": self._options.get("verbose", False),
            "idf": self._options.get("idf", False),
            "batch_size": self._options.get("batch_size", 64), # bert_score's internal batch_size
            "device": self._options.get("device", None) # Let bert_score auto-detect or specify
        }
        if model_type: # Only add if specified, otherwise library default.
            bertscore_options["model_type"] = model_type
        
        try:
            P, R, F1 = bert_score.score(
                cands=self._collected_predictions,
                refs=self._collected_labels,
                **bertscore_options
            )
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item()
            }
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0, "error": str(e)}


class HFcommonPipelineMetric(BaseMetric): # Common base for HF pipeline based metrics
    """Base for metrics using Hugging Face pipelines, computes average score."""
    def __init__(self):
        super().__init__()
        self.scores_sum = 0.0
        self.count = 0
        self._pipeline = None
        self._pipeline_task_name: str = "" # e.g. "question-answering"
        self._pipeline_model_name: str = "" # e.g. "valhalla/squad-small-finetuned-bert-base-uncased"
        self._score_key_in_result: str = "score" # Key to extract score from pipeline output
        self._result_if_no_data: Any = 0.0


    def _initialize_pipeline(self):
        if self._pipeline is None:
            try:
                # Get model name from options, fallback to a default if specified in class
                model_name_from_options = self._options.get("model", self._pipeline_model_name)
                if not model_name_from_options:
                    logger.error(f"No model specified for {self.__class__.__name__} pipeline.")
                    return False
                
                self._pipeline = pipeline(self._pipeline_task_name, model=model_name_from_options, device=-1) # device=-1 for CPU, or configure
                logger.info(f"Initialized {self._pipeline_task_name} pipeline with model {model_name_from_options} for {self.__class__.__name__}")
                return True
            except Exception as e:
                logger.error(f"Failed to load HF pipeline for {self.__class__.__name__} (task: {self._pipeline_task_name}, model: {model_name_from_options}): {e}")
                return False
        return True


    def reset_state(self) -> None:
        self.scores_sum = 0.0
        self.count = 0
        # self._pipeline = None # Avoid re-initializing pipeline on every task run unless config changes

    def _process_item(self, prediction: Any, label: Any) -> Union[float, None]:
        """
        Process a single prediction/label pair using the HF pipeline.
        To be implemented by subclasses.
        Should return the score for the item, or None if failed.
        """
        raise NotImplementedError("Subclasses must implement _process_item.")

    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        if not self._initialize_pipeline() or self._pipeline is None:
            logger.warning(f"Pipeline for {self.__class__.__name__} not available. Skipping update.")
            return

        for pred_item, label_item in zip(predictions, labels):
            try:
                score = self._process_item(pred_item, label_item)
                if score is not None:
                    self.scores_sum += score
                    self.count += 1
            except Exception as e:
                logger.error(f"Error processing item in {self.__class__.__name__} for pred '{str(pred_item)[:50]}': {e}", exc_info=False)


    def result(self) -> Any: # Can be float or Dict
        if self.count == 0:
            return self._result_if_no_data
        return self.scores_sum / self.count


class FactualConsistencyMetric(HFcommonPipelineMetric):
    """Computes factual consistency between answer and context (label)."""
    def __init__(self):
        super().__init__()
        self._pipeline_task_name = "question-answering"
        # Default model, can be overridden by "model" in options
        self._pipeline_model_name = self._options.get("model", "deepset/tinyroberta-squad2") # or a larger one
        self._score_key_in_result = "score" # QA pipeline returns 'score'
        # Factual consistency might be binary (consistent/not) based on threshold, or average confidence.
        # Here, we'll average the QA model's confidence score for the answer being found in the context.

    def _process_item(self, prediction_answer: str, label_context: str) -> Union[float, None]:
        if not self._pipeline: return None
        # Prediction is the answer, Label is the context
        try:
            # The "question" is the generated answer, "context" is the source document/label
            qa_input = {"question": str(prediction_answer), "context": str(label_context)}
            # max_answer_len=100 # from original, maybe not needed if we just want confidence
            result = self._pipeline(qa_input, max_answer_len=len(str(prediction_answer)) + 20) # Give some leeway
            
            # Interpretation: if the answer (question) is found in the context with high confidence,
            # it's considered consistent.
            # A more robust factual consistency might involve NLI models.
            # This is a proxy using QA.
            # We use the score directly as a measure of consistency.
            # Original used a threshold: 1.0 if result['score'] > 0.5 else 0.0
            # Let's stick to averaging the score for now, or make threshold configurable.
            threshold = self._options.get("consistency_threshold", None)
            if threshold is not None:
                return 1.0 if result[self._score_key_in_result] >= threshold else 0.0
            return result[self._score_key_in_result] 
        except Exception as e:
            # Log specific error for QA failure
            logger.debug(f"FactualConsistencyMetric: QA pipeline error for q='{str(prediction_answer)[:50]}', c='{str(label_context)[:50]}': {e}")
            return None # Or 0.0 if preferred on error


class ToxicityScoreMetric(HFcommonPipelineMetric):
    """Computes toxicity score using pre-trained classifier."""
    def __init__(self):
        super().__init__()
        self._pipeline_task_name = "text-classification"
        # Default model, can be overridden by "model" in options
        self._pipeline_model_name = self._options.get("model", "unitary/toxic-bert")
        # This pipeline can return multiple labels (TOXICITY, SEVERE_TOXICITY, etc.)
        # We need to specify which score to average or return a dict.
        # For simplicity, let's average the score for the 'TOXICITY' label.
        self._target_label_for_score = self._options.get("target_label", "TOXICITY").upper()
        self._result_if_no_data = {"toxicity": 0.0} # Default dict structure

    def _process_item(self, prediction_text: str, label: Any) -> Union[float, None]: # Label is ignored
        if not self._pipeline: return None
        try:
            # The pipeline might return a list of dicts if top_k=None (which it is by default for this task)
            results = self._pipeline(str(prediction_text), top_k=None) # Get all label scores
            if results and isinstance(results, list) and isinstance(results[0], list): # Check for nested list output
                results = results[0] # Take the first list if pipeline wraps output

            for res_item in results:
                if res_item['label'].upper() == self._target_label_for_score:
                    return res_item['score']
            logger.warning(f"ToxicityScoreMetric: Target label '{self._target_label_for_score}' not found in pipeline results: {results}")
            return None # Target label not found
        except Exception as e:
            logger.debug(f"ToxicityScoreMetric: Pipeline error for text '{str(prediction_text)[:50]}': {e}")
            return None

    def result(self) -> Dict[str, float]: # Overriding to return a dict
        # The original returned a dict with "toxicity" and "severe_toxicity"
        # This refactored one (using HFcommonPipelineMetric) averages one target label.
        # To match original, this would need to be more custom.
        # For now, it returns average of self._target_label_for_score
        avg_score = super().result() # This will be a float (average of the target label)
        return {self._target_label_for_score.lower(): avg_score}