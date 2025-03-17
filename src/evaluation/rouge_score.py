from typing import Any
from evaluation.base_model_evaluator import BaseMetric
from rouge import Rouge

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
        rouge = Rouge()
        scores = rouge.get_scores(predictions, labels)
        return scores[0]["rouge-l"]["f"]  # Return F1 score for ROUGE-L