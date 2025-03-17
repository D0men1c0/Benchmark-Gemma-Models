from typing import Any, Dict, List

from evaluation.metric_factory import MetricFactory

class Evaluator:
    """
    Evaluates benchmark results based on the specified metrics.

    :param evaluation_params: Parameters like batch size, etc.
    """
    def __init__(self, evaluation_params: Dict[str, Any]):
        self.evaluation_params = evaluation_params

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

        for metric_config in metrics:
            metric_name = metric_config["name"]
            metric_options = metric_config.get("options", {})
            try:
                metric = MetricFactory.get_metric(metric_name)
                metric_value = metric.compute(predictions, labels, **metric_options)
                evaluation_results[metric_name] = metric_value
            except ValueError as e:
                print(f"Error computing metric {metric_name}: {e}")

        return evaluation_results