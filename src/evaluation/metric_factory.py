from evaluation.accuracy import AccuracyMetric
from evaluation.base_model_evaluator import BaseMetric
from evaluation.bleu_score import BLEUScoreMetric
from evaluation.exact_match import ExactMatchMetric
from evaluation.f1_score import F1ScoreMetric
from evaluation.perplexity import PerplexityMetric
from evaluation.precision import PrecisionMetric
from evaluation.recall import RecallMetric
from evaluation.rouge_score import ROUGEScoreMetric


class MetricFactory:
    """
    Factory class to create evaluation metrics.
    """
    @staticmethod
    def get_metric(metric_name: str) -> BaseMetric:
        """
        Get the metric instance based on the metric name.

        :param metric_name: Name of the metric (e.g., "accuracy", "f1_score").
        :return: An instance of the metric.
        :raises ValueError: If the metric is not supported.
        """
        if metric_name == "accuracy":
            return AccuracyMetric()
        elif metric_name == "precision":
            return PrecisionMetric()
        elif metric_name == "recall":
            return RecallMetric()
        elif metric_name == "f1_score":
            return F1ScoreMetric()
        elif metric_name == "perplexity":
            return PerplexityMetric()
        elif metric_name == "exact_match":
            return ExactMatchMetric()
        elif metric_name == "bleu":
            return BLEUScoreMetric()
        elif metric_name == "rouge":
            return ROUGEScoreMetric()
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")