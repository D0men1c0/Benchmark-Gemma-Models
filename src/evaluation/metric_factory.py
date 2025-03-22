from typing import List
from torch import Type
from .base_metrics import BaseMetric
from .concrete_metrics import AccuracyMetric, PrecisionMetric, RecallMetric, F1ScoreMetric, PerplexityMetric, ExactMatchMetric, BLEUScoreMetric, ROUGEScoreMetric

class MetricFactory:
    """
    Factory class to create evaluation metrics using dictionary mapping.
    """
    _METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
        "accuracy": AccuracyMetric,
        "precision": PrecisionMetric,
        "recall": RecallMetric,
        "f1_score": F1ScoreMetric,
        "perplexity": PerplexityMetric,
        "exact_match": ExactMatchMetric,
        "bleu": BLEUScoreMetric,
        "rouge": ROUGEScoreMetric,
    }

    @classmethod
    def register_metric(cls, name: str, metric_class: Type[BaseMetric]):
        """Allow dynamic registration of new metrics"""
        cls._METRIC_REGISTRY[name.lower()] = metric_class

    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of registered metrics"""
        return list(cls._METRIC_REGISTRY.keys())

    @classmethod
    def get_metric(cls, metric_name: str) -> BaseMetric:
        """
        Get the metric instance using dictionary lookup.
        
        :param metric_name: Name of the metric to instantiate
        :return: Metric instance
        :raises ValueError: For unsupported metrics
        """
        metric_class = cls._METRIC_REGISTRY.get(metric_name.lower())
        if not metric_class:
            available = ", ".join(cls._METRIC_REGISTRY.keys())
            raise ValueError(
                f"Unsupported metric: {metric_name}. "
                f"Available metrics: {available}"
            )
        return metric_class()