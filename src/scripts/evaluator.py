from typing import Any, Dict, List
from evaluation.metric_factory import MetricFactory
from utils.logger import setup_logger

class Evaluator:
    """
    Evaluates benchmark results based on the specified metrics.

    :param evaluation_params: Parameters like batch size, etc.
    """
    def __init__(self, evaluation_params: Dict[str, Any]):
        self.evaluation_params = evaluation_params
        self.logger = setup_logger()

    def evaluate(self, task_results: Dict[str, Any], metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the task results using the specified metrics.

        :param task_results: The results obtained from running a task.
        :param metrics: List of metric configurations (e.g., [{"name": "accuracy"}, {"name": "f1_score", "average": "macro"}]).
        :return: Dictionary of computed metric values.
        """
        evaluation_results = {}
        predictions = task_results.get("predictions")
        labels = task_results.get("labels")

        if not predictions or not labels:
            raise ValueError("Missing predictions/labels for evaluation")
        if len(predictions) != len(labels):
            self.logger.warning("Predictions/labels length mismatch")

        for metric_config in metrics:
            metric_name = metric_config["name"]
            metric_options = metric_config.get("options", {})
            try:
                metric = MetricFactory.get_metric(metric_name)
                metric_value = metric.compute(predictions, labels, **metric_options)
                evaluation_results[metric_name] = metric_value
            except Exception as e:
                self.logger.error(f"Metric {metric_name} failed: {str(e)}")
                raise

        return evaluation_results