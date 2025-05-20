from typing import Any, Dict, List, Optional, Tuple
from benchmark.evaluation.metrics.base_metrics import BaseMetric
from .metrics.metric_factory import MetricFactory
from utils.logger import setup_logger

class Evaluator:
    """
    Evaluates benchmark results based on the specified metrics using batched updates.
    """
    def __init__(self, evaluation_params: Dict[str, Any]): # evaluation_params not used in this snippet, but keep for consistency
        """
        Initialize the Evaluator with evaluation parameters.
        :param evaluation_params: Dictionary of evaluation parameters.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self._metric_instances_with_config: List[Tuple[Dict[str, Any], BaseMetric]] = []
        self._metrics_configs: List[Dict[str, Any]] = []

    def prepare_metrics(self, metrics_config: List[Dict[str, Any]]):
        """
        Initialize and reset metric instances based on the configuration.
        :param metrics_config: List of metric configurations.
        """
        self._metric_instances_with_config = []
        self._metrics_configs = metrics_config
        for metric_conf in self._metrics_configs:
            metric_name_from_yaml = metric_conf.get("name")
            if not metric_name_from_yaml:
                self.logger.error(f"Metric configuration missing 'name': {metric_conf}. Skipping this metric.")
                continue
            
            current_metric_options = metric_conf.get("options", {})

            try:
                metric_instance = MetricFactory.get_metric(metric_name_from_yaml)

                metric_instance.set_options(**current_metric_options)
                metric_instance.reset_state()
                
                self._metric_instances_with_config.append((metric_conf, metric_instance))
                self.logger.debug(f"Metric '{metric_name_from_yaml}' (options: {current_metric_options}) initialized and reset.")
            except Exception as e:
                self.logger.error(f"Failed to initialize metric '{metric_name_from_yaml}': {str(e)}", exc_info=True)

    def update_batch_metrics(self, batch_outputs: Dict[str, List[Any]], task_name: Optional[str] = None):
        """
        Update all prepared metrics with a new batch of predictions and labels.
        :param batch_outputs: Dictionary containing predictions and labels, or a tuple.
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
            self.logger.warning(f"Evaluator.update_batch_metrics received only a list (predictions) for task '{task_name}'. Labels will be considered empty for metrics requiring them.")
            default_labels = [None] * len(default_predictions)
        else:
            self.logger.error(f"Evaluator.update_batch_metrics received unexpected batch_outputs type: {type(batch_outputs)} for task '{task_name}'")
            return

        for metric_conf, metric_instance in self._metric_instances_with_config:
            metric_name_from_yaml = metric_conf.get("name", "unknown_metric")
            current_predictions_for_metric = None
            current_labels_for_metric = None

            if isinstance(batch_outputs, dict) and task_name and "GSM8K" in task_name.upper():
                if metric_name_from_yaml == "exact_match":
                    current_predictions_for_metric = batch_outputs.get("exact_match_predictions")
                    current_labels_for_metric = batch_outputs.get("labels_for_exact_match")
                else:
                    current_predictions_for_metric = batch_outputs.get("text_predictions")
                    current_labels_for_metric = batch_outputs.get("labels_for_text")
            else:
                current_predictions_for_metric = default_predictions
                current_labels_for_metric = default_labels
            
            if current_predictions_for_metric is None or current_labels_for_metric is None:
                self.logger.warning(
                    f"Metric '{metric_name_from_yaml}': Predictions or labels are missing for task '{task_name}'. Skipping update for this metric."
                )
                continue
            
            if not isinstance(current_predictions_for_metric, list) or not isinstance(current_labels_for_metric, list):
                self.logger.warning(
                    f"Metric '{metric_name_from_yaml}': Predictions or labels are not lists for task '{task_name}'. Skipping update."
                )
                continue
            
            if len(current_predictions_for_metric) != len(current_labels_for_metric):
                self.logger.warning(
                    f"Metric '{metric_name_from_yaml}': Mismatch between #predictions ({len(current_predictions_for_metric)}) and "
                    f"#labels ({len(current_labels_for_metric)}) for task '{task_name}'. Skipping update for this metric."
                )
                continue

            try:
                metric_instance.update_state(current_predictions_for_metric, current_labels_for_metric)
            except Exception as e:
                self.logger.error(f"Error updating metric {metric_name_from_yaml} for batch: {str(e)} (Task: {task_name})", exc_info=True)

    def finalize_results(self) -> Dict[str, Any]:
        """
        Compute final results for all metrics.
        :return: Dictionary of computed metric values.
        """
        evaluation_results: Dict[str, Any] = {}
        
        for metric_conf, metric_instance in self._metric_instances_with_config:
            metric_name_from_yaml = metric_conf.get("name")
            options = metric_conf.get("options", {})
            
            if not metric_name_from_yaml:
                continue

            try:
                result = metric_instance.result()
                if metric_name_from_yaml == "custom_script" and isinstance(result, dict):
                    for key, value in result.items():
                        if key in evaluation_results:
                            self.logger.warning(
                                f"Metric key collision in final results: '{key}' generated by a custom_script "
                                f"(options: {options}) is overwriting an existing key. "
                                "Ensure custom metric result keys (e.g., from 'result_key_name' in script_args) are unique."
                            )
                        evaluation_results[key] = value
                else:
                    if metric_name_from_yaml in evaluation_results:
                         self.logger.warning(
                            f"Metric name collision in final results: '{metric_name_from_yaml}' "
                            f"(options: {options}) is overwriting an existing entry with the same name."
                        )
                    evaluation_results[metric_name_from_yaml] = result
                
                self.logger.info(f"Finalized metric '{metric_name_from_yaml}' (options: {options}): {result}")
            except Exception as e:
                self.logger.error(f"Error finalizing result for metric {metric_name_from_yaml} (options: {options}): {str(e)}", exc_info=True)
                error_key = f"{metric_name_from_yaml}_error"
                if options:
                    error_key = f"{metric_name_from_yaml}_{str(options.get('metric_script_args', {}).get('result_key_name', 'opts'))}_error"
                evaluation_results[error_key] = f"Failed to compute: {str(e)}"
                
        return evaluation_results