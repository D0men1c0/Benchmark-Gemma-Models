from typing import Any
from nltk.translate.bleu_score import sentence_bleu

from evaluation.base_model_evaluator import BaseMetric

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
        weights = kwargs.get("weights", (0.25, 0.25, 0.25, 0.25))  # Default to 4-gram BLEU
        return sentence_bleu([labels], predictions, weights=weights)