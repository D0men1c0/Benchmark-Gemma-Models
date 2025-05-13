from typing import Any, Dict, List
from benchmark.evaluation.metrics.base_metrics import BaseMetric
from .metrics.metric_factory import MetricFactory
from utils.logger import setup_logger

class Evaluator:
    """
    Evaluates benchmark results based on the specified metrics using batched updates.
    """
    def __init__(self, evaluation_params: Dict[str, Any]): # evaluation_params not used in this snippet, but keep for consistency
        self.logger = setup_logger(__name__)
        self._metrics_instances: Dict[str, BaseMetric] = {}
        self._metrics_configs: List[Dict[str, Any]] = []

    def prepare_metrics(self, metrics_config: List[Dict[str, Any]]):
        """
        Initialize and reset metric instances based on the configuration.
        :param metrics_config: List of metric configurations.
        """
        self._metrics_instances = {}
        self._metrics_configs = metrics_config
        for metric_conf in self._metrics_configs:
            metric_name = metric_conf["name"]
            try:
                metric_instance = MetricFactory.get_metric(metric_name)
                metric_instance.reset_state() # Ensure it's clean
                self._metrics_instances[metric_name] = metric_instance
                self.logger.debug(f"Metric '{metric_name}' initialized and reset.")
            except Exception as e:
                self.logger.error(f"Failed to initialize metric {metric_name}: {str(e)}")
                # Decide if this should raise an error or just skip the metric
                # For now, it will be missing from _metrics_instances

    def update_batch_metrics(self, batch_predictions: List[Any], batch_labels: List[Any]):
        """
        Update all prepared metrics with a new batch of predictions and labels.
        :param batch_predictions: Predictions for the current batch.
        :param batch_labels: Labels for the current batch.
        """
        if not batch_predictions or not batch_labels:
            self.logger.warning("Empty predictions or labels passed to update_batch_metrics. Skipping update.")
            return

        for metric_conf in self._metrics_configs:
            metric_name = metric_conf["name"]
            metric_options = metric_conf.get("options", {})
            
            if metric_name in self._metrics_instances:
                try:
                    metric_instance = self._metrics_instances[metric_name]
                    metric_instance.update_state(batch_predictions, batch_labels, **metric_options)
                except Exception as e:
                    self.logger.error(f"Error updating metric {metric_name} for batch: {str(e)}", exc_info=True)
                    # Optionally remove the metric or mark as failed
            else:
                self.logger.warning(f"Metric {metric_name} was not initialized. Skipping update.")


    def finalize_results(self) -> Dict[str, Any]:
        """
        Compute final results for all metrics.
        :return: Dictionary of computed metric values.
        """
        evaluation_results = {}
        for metric_conf in self._metrics_configs:
            metric_name = metric_conf["name"]
            if metric_name in self._metrics_instances:
                try:
                    metric_instance = self._metrics_instances[metric_name]
                    evaluation_results[metric_name] = metric_instance.result()
                    self.logger.info(f"Finalized metric '{metric_name}': {evaluation_results[metric_name]}")
                except Exception as e:
                    self.logger.error(f"Error finalizing result for metric {metric_name}: {str(e)}", exc_info=True)
                    evaluation_results[metric_name] = {"error": f"Failed to compute: {str(e)}"}
            else:
                 evaluation_results[metric_name] = {"error": "Metric not initialized or failed during init."}
        return evaluation_results