from typing import Any, Dict, List, Optional
from benchmark.evaluation.metrics.base_metrics import BaseMetric
from .metrics.metric_factory import MetricFactory
from utils.logger import setup_logger

class Evaluator:
    """
    Evaluates benchmark results based on the specified metrics using batched updates.
    """
    def __init__(self, evaluation_params: Dict[str, Any]): # evaluation_params not used in this snippet, but keep for consistency
        self.logger = setup_logger(self.__class__.__name__)
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
                current_metric_options = metric_conf.get("options", {}) # Default to empty dict
                metric_instance.set_options(**current_metric_options)
                self.logger.debug(f"Metric '{metric_name}': Set options to {current_metric_options}")
                
                # 2. Then reset state (this will call init_fn which should now be loaded)
                metric_instance.reset_state() 
                self._metrics_instances[metric_name] = metric_instance
                self.logger.debug(f"Metric '{metric_name}' initialized and reset.")
            except Exception as e:
                self.logger.error(f"Failed to initialize metric {metric_name}: {str(e)}")
                # Decide if this should raise an error or just skip the metric
                # For now, it will be missing from _metrics_instances

    def update_batch_metrics(self, batch_outputs: Dict[str, List[Any]], task_name: Optional[str] = None):
        """
        Update all prepared metrics with a new batch of predictions and labels.
        batch_outputs can now be a dictionary for tasks like GSM8K.
        :param batch_outputs: Dictionary containing predictions and labels.
        :param task_name: Optional task name for specific handling.
        """
        default_predictions = batch_outputs 
        default_labels = []

        if isinstance(batch_outputs, dict) and "labels_for_text" in batch_outputs:
            pass
        elif isinstance(batch_outputs, tuple) and len(batch_outputs) == 2:
            default_predictions = batch_outputs[0]
            default_labels = batch_outputs[1]
        elif isinstance(batch_outputs, list):
            self.logger.error("Evaluator.update_batch_metrics received only a list (predictions) but also needs labels.")
            return
        else:
            self.logger.error(f"Evaluator.update_batch_metrics received unexpected batch_outputs type: {type(batch_outputs)}")
            return


        for metric_name, metric_instance in self._metrics_instances.items():
            current_predictions = None
            current_labels = None

            if isinstance(batch_outputs, dict) and task_name and "GSM8K" in task_name:
                if metric_name == "exact_match":
                    current_predictions = batch_outputs.get("exact_match_predictions")
                    current_labels = batch_outputs.get("labels_for_exact_match")
                else:
                    current_predictions = batch_outputs.get("text_predictions")
                    current_labels = batch_outputs.get("labels_for_text")
            else:
                if not default_labels and not isinstance(default_predictions, dict):
                    self.logger.warning(f"Metric '{metric_name}': Default labels are missing for non-dict batch_outputs. Skipping update.")
                    continue

                current_predictions = default_predictions
                current_labels = default_labels

            if current_predictions is None or current_labels is None:
                self.logger.warning(
                    f"Metric '{metric_name}': Predictions or labels are missing for task '{task_name}'. Skipping update."
                )
                continue
            
            if not isinstance(current_predictions, list) or not isinstance(current_labels, list):
                self.logger.warning(
                    f"Metric '{metric_name}': Predictions or labels are not lists for task '{task_name}'. Skipping update."
                )
                continue

            try:
                metric_instance.update_state(current_predictions, current_labels)
            except Exception as e:
                self.logger.error(f"Error updating metric {metric_name} for batch: {str(e)} (Task: {task_name})", exc_info=True)


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