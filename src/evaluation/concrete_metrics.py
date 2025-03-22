from typing import Any
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np
from .base_metrics import BaseMetric
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

class AccuracyMetric(BaseMetric):
    """
    Class for computing accuracy.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute accuracy.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., top_k).
        :return: Accuracy score.
        """
        top_k = kwargs.get("top_k", None)
        if top_k:
            raise NotImplementedError("Top-k accuracy not implemented")
        return accuracy_score(labels, predictions)

class PrecisionMetric(BaseMetric):
    """
    Class for computing precision.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute precision.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: Precision score.
        """
        average = kwargs.get("average", "weighted")
        return precision_score(labels, predictions, average=average)

class F1ScoreMetric(BaseMetric):
    """
    Class for computing F1 score.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute F1 score.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: F1 score.
        """
        average = kwargs.get("average", "weighted")
        return f1_score(labels, predictions, average=average)

class RecallMetric(BaseMetric):
    """
    Class for computing recall.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute recall.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: Recall score.
        """
        average = kwargs.get("average", "weighted")
        return recall_score(labels, predictions, average=average)

class BLEUScoreMetric(BaseMetric):
    """
    Class for computing BLEU score.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute BLEU score.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., weights).
        :return: BLEU score.
        """
        if not isinstance(labels, list) or not isinstance(predictions, list):
            raise TypeError("BLEU requires list of reference sentences")
        weights = kwargs.get("weights", (0.25, 0.25, 0.25, 0.25))  # Default to 4-gram BLEU
        return sentence_bleu([labels], predictions, weights=weights)

class ROUGEScoreMetric(BaseMetric):
    """
    Class for computing ROUGE score.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute ROUGE score.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., metrics).
        :return: ROUGE score.
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have same length")
        rouge = Rouge()
        scores = rouge.get_scores(predictions, labels)
        return scores[0]["rouge-l"]["f"]  # Return F1 score for ROUGE-L
    
class PerplexityMetric(BaseMetric):
    """
    Class for computing perplexity.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute perplexity.

        :param predictions: Model predictions (probabilities, not text).
        :param labels: Ground truth labels.
        :param kwargs: Additional options.
        :return: Perplexity score.
        """
        predictions = np.array(predictions, dtype=np.float32)

        # Ensure probabilities are in (0, 1] to avoid log(0) errors
        predictions = np.clip(predictions, 1e-10, 1.0)

        log_probs = np.log(predictions)
        perplexity = np.exp(-np.mean(log_probs))

        return perplexity

class ExactMatchMetric(BaseMetric):
    """
    Class for computing exact match.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute exact match.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options.
        :return: Exact match score.
        """
        return float(predictions == labels)