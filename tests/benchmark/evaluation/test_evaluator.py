import pytest
from unittest.mock import patch, MagicMock, call

from src.benchmark.evaluation.evaluator import Evaluator
from src.benchmark.evaluation.metrics.base_metrics import BaseMetric
from src.benchmark.evaluation.metrics.metric_factory import MetricFactory
from src.config_models import MetricConfig


@pytest.fixture
def mock_metric_config_accuracy():
    return {"name": "accuracy", "options": {}}

@pytest.fixture
def mock_metric_config_rouge():
    return {"name": "rouge", "options": {"metrics": ["rougeL"], "stats": ["f"]}}

@pytest.fixture
def mock_metric_config_custom_script(tmp_path):
    # Create a dummy metric script for testing CustomScriptMetric via Evaluator
    script_content = """
def init_state(options): return {"count": 0, "total": 0.0, "key_name": options.get("result_key_name", "custom_score")}
def update_state(state, predictions, labels, options):
    for p, l in zip(predictions, labels): state["total"] += p * l; state["count"] += 1
    return state
def get_results(state, options): return {state["key_name"]: state["total"] / state["count"] if state["count"] > 0 else 0}
"""
    script_file = tmp_path / "dummy_metric_script.py"
    script_file.write_text(script_content)
    return {
        "name": "custom_script",
        "options": {
            "metric_script_path": str(script_file),
            "metric_init_function_name": "init_state",
            "metric_update_function_name": "update_state",
            "metric_result_function_name": "get_results",
            "metric_script_args": {"result_key_name": "my_custom_metric_score"}
        }
    }

@pytest.fixture
def dummy_evaluation_params():
    # This is passed to Evaluator constructor, but not directly used in current SUT methods.
    # Kept for potential future use or if SUT changes.
    return {"some_eval_param": "value"} 

@patch('src.benchmark.evaluation.evaluator.setup_logger')
def test_evaluator_init(mock_setup_logger, dummy_evaluation_params):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
    assert evaluator.logger is mock_logger
    assert evaluator._metric_instances_with_config == []
    assert evaluator._metrics_configs == []

@patch('src.benchmark.evaluation.evaluator.setup_logger')
@patch('src.benchmark.evaluation.evaluator.MetricFactory.get_metric')
def test_evaluator_prepare_metrics_success(
    mock_get_metric, mock_setup_logger, dummy_evaluation_params, 
    mock_metric_config_accuracy, mock_metric_config_rouge
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    mock_accuracy_metric_instance = MagicMock(spec=BaseMetric)
    mock_rouge_metric_instance = MagicMock(spec=BaseMetric)
    mock_get_metric.side_effect = [mock_accuracy_metric_instance, mock_rouge_metric_instance]

    evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
    metrics_config = [mock_metric_config_accuracy, mock_metric_config_rouge]
    evaluator.prepare_metrics(metrics_config)

    assert len(evaluator._metric_instances_with_config) == 2
    assert evaluator._metrics_configs == metrics_config

    # Check for accuracy metric
    acc_conf, acc_inst = evaluator._metric_instances_with_config[0]
    assert acc_conf == mock_metric_config_accuracy
    assert acc_inst is mock_accuracy_metric_instance
    mock_accuracy_metric_instance.set_options.assert_called_once_with(**mock_metric_config_accuracy["options"])
    mock_accuracy_metric_instance.reset_state.assert_called_once()

    # Check for rouge metric
    rouge_conf, rouge_inst = evaluator._metric_instances_with_config[1]
    assert rouge_conf == mock_metric_config_rouge
    assert rouge_inst is mock_rouge_metric_instance
    mock_rouge_metric_instance.set_options.assert_called_once_with(**mock_metric_config_rouge["options"])
    mock_rouge_metric_instance.reset_state.assert_called_once()

    mock_logger.debug.assert_any_call("Metric 'accuracy' (options: {}) initialized and reset.")
    mock_logger.debug.assert_any_call("Metric 'rouge' (options: {'metrics': ['rougeL'], 'stats': ['f']}) initialized and reset.")

@patch('src.benchmark.evaluation.evaluator.setup_logger')
@patch('src.benchmark.evaluation.evaluator.MetricFactory.get_metric')
def test_evaluator_prepare_metrics_missing_name_in_config(
    mock_get_metric, mock_setup_logger, dummy_evaluation_params
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
    invalid_metric_config = [{"options": {}}] # Missing "name"
    evaluator.prepare_metrics(invalid_metric_config)

    assert len(evaluator._metric_instances_with_config) == 0
    mock_logger.error.assert_called_with(f"Metric configuration missing 'name': {invalid_metric_config[0]}. Skipping this metric.")
    mock_get_metric.assert_not_called()

@patch('src.benchmark.evaluation.evaluator.setup_logger')
@patch('src.benchmark.evaluation.evaluator.MetricFactory.get_metric', side_effect=ValueError("Unsupported metric"))
def test_evaluator_prepare_metrics_factory_fails(
    mock_get_metric, mock_setup_logger, dummy_evaluation_params, mock_metric_config_accuracy
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
    evaluator.prepare_metrics([mock_metric_config_accuracy])

    assert len(evaluator._metric_instances_with_config) == 0
    mock_logger.error.assert_called_with(
        "Failed to initialize metric 'accuracy': Unsupported metric", exc_info=True
    )

@patch('src.benchmark.evaluation.evaluator.setup_logger')
def test_evaluator_update_batch_metrics_tuple_input(
    mock_setup_logger, dummy_evaluation_params, mock_metric_config_accuracy
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    mock_metric_instance = MagicMock(spec=BaseMetric)
    with patch.object(MetricFactory, 'get_metric', return_value=mock_metric_instance):
        evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
        evaluator.prepare_metrics([mock_metric_config_accuracy])

    batch_outputs_tuple = (["pred1", "pred2"], ["label1", "label2"])
    evaluator.update_batch_metrics(batch_outputs_tuple, task_name="test_task")

    mock_metric_instance.update_state.assert_called_once_with(["pred1", "pred2"], ["label1", "label2"])

@patch('src.benchmark.evaluation.evaluator.setup_logger')
def test_evaluator_update_batch_metrics_dict_input_gsm8k(
    mock_setup_logger, dummy_evaluation_params, mock_metric_config_accuracy, mock_metric_config_rouge
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    mock_accuracy_instance = MagicMock(spec=BaseMetric, name="AccuracyMock")
    mock_rouge_instance = MagicMock(spec=BaseMetric, name="RougeMock")
    
    def get_metric_side_effect(metric_name):
        if metric_name == "accuracy": return mock_accuracy_instance
        if metric_name == "rouge": return mock_rouge_instance
        return MagicMock(spec=BaseMetric)

    with patch.object(MetricFactory, 'get_metric', side_effect=get_metric_side_effect):
        evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
        # Prepare with exact_match (mapped to accuracy mock for simplicity here) and rouge
        metrics_config = [
            {"name": "exact_match", "options": {}}, # Will use accuracy mock
            mock_metric_config_rouge
        ]
        evaluator.prepare_metrics(metrics_config) 
        # Rename the first metric's name to 'exact_match' for the assertion below.
        # This is because the test setup for MetricFactory uses "accuracy" for the first mock.
        # A bit of a hack, cleaner would be to have separate mocks per metric name if needed.
        evaluator._metric_instances_with_config[0] = (metrics_config[0], mock_accuracy_instance)


    batch_outputs_dict = {
        "text_predictions": ["full text pred1", "full text pred2"],
        "labels_for_text": ["full text label1", "full text label2"],
        "exact_match_predictions": ["4", "3"],
        "labels_for_exact_match": ["4", "3"]
    }
    evaluator.update_batch_metrics(batch_outputs_dict, task_name="GSM8K_task")

    # Exact match metric should use specific keys
    mock_accuracy_instance.update_state.assert_called_once_with(["4", "3"], ["4", "3"])
    # Other metrics (like ROUGE) should use text_predictions and labels_for_text
    mock_rouge_instance.update_state.assert_called_once_with(
        ["full text pred1", "full text pred2"], ["full text label1", "full text label2"]
    )

@patch('src.benchmark.evaluation.evaluator.setup_logger')
def test_evaluator_update_batch_metrics_mismatch_length(
    mock_setup_logger, dummy_evaluation_params, mock_metric_config_accuracy
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    mock_metric_instance = MagicMock(spec=BaseMetric)
    with patch.object(MetricFactory, 'get_metric', return_value=mock_metric_instance):
        evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
        evaluator.prepare_metrics([mock_metric_config_accuracy])

    batch_outputs_mismatch = (["pred1"], ["label1", "label2"])
    evaluator.update_batch_metrics(batch_outputs_mismatch, task_name="mismatch_task")

    mock_logger.warning.assert_called_with(
        "Metric 'accuracy': Mismatch between #predictions (1) and #labels (2) for task 'mismatch_task'. Skipping update for this metric."
    )
    mock_metric_instance.update_state.assert_not_called()

@patch('src.benchmark.evaluation.evaluator.setup_logger')
def test_evaluator_finalize_results_success(
    mock_setup_logger, dummy_evaluation_params, 
    mock_metric_config_accuracy, mock_metric_config_rouge
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    mock_accuracy_metric = MagicMock(spec=BaseMetric)
    mock_accuracy_metric.result.return_value = 0.95
    mock_rouge_metric = MagicMock(spec=BaseMetric)
    mock_rouge_metric.result.return_value = {"rougeL_f": 0.75}

    def get_metric_side_effect(metric_name):
        if metric_name == "accuracy": return mock_accuracy_metric
        if metric_name == "rouge": return mock_rouge_metric
        return MagicMock(spec=BaseMetric)

    with patch.object(MetricFactory, 'get_metric', side_effect=get_metric_side_effect):
        evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
        evaluator.prepare_metrics([mock_metric_config_accuracy, mock_metric_config_rouge])
    
    results = evaluator.finalize_results()

    assert results == {"accuracy": 0.95, "rouge": {"rougeL_f": 0.75}}
    mock_accuracy_metric.result.assert_called_once()
    mock_rouge_metric.result.assert_called_once()
    mock_logger.info.assert_any_call("Finalized metric 'accuracy' (options: {}): 0.95")
    mock_logger.info.assert_any_call("Finalized metric 'rouge' (options: {'metrics': ['rougeL'], 'stats': ['f']}): {'rougeL_f': 0.75}")

@patch('src.benchmark.evaluation.evaluator.setup_logger')
def test_evaluator_finalize_results_metric_result_fails(
    mock_setup_logger, dummy_evaluation_params, mock_metric_config_accuracy
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    
    mock_metric_instance = MagicMock(spec=BaseMetric)
    mock_metric_instance.result.side_effect = Exception("Metric computation error")

    with patch.object(MetricFactory, 'get_metric', return_value=mock_metric_instance):
        evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
        evaluator.prepare_metrics([mock_metric_config_accuracy])
        
    results = evaluator.finalize_results()

    assert "accuracy_error" in results # or similar key based on options
    assert "Failed to compute: Metric computation error" in results["accuracy_error"]
    mock_logger.error.assert_called_with(
        "Error finalizing result for metric accuracy (options: {}): Metric computation error", exc_info=True
    )

@patch('src.benchmark.evaluation.evaluator.setup_logger')
@patch('src.benchmark.evaluation.evaluator.MetricFactory.get_metric')
def test_evaluator_finalize_results_custom_script_metric_dict_output(
    mock_get_metric, mock_setup_logger, dummy_evaluation_params, mock_metric_config_custom_script
):
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger

    # We need a real CustomScriptMetric or a mock that behaves like one for .result()
    # For simplicity, let's mock its .result() directly
    mock_custom_metric_instance = MagicMock(spec=BaseMetric) # It's BaseMetric but will return a dict
    custom_result_dict = {"my_custom_metric_score": 0.88, "another_custom": 0.77}
    mock_custom_metric_instance.result.return_value = custom_result_dict
    mock_get_metric.return_value = mock_custom_metric_instance
    
    evaluator = Evaluator(evaluation_params=dummy_evaluation_params)
    evaluator.prepare_metrics([mock_metric_config_custom_script])
    results = evaluator.finalize_results()

    assert results == custom_result_dict # The custom metric's dict keys should be directly in the results
    mock_logger.info.assert_any_call(
        f"Finalized metric 'custom_script' (options: {mock_metric_config_custom_script['options']}): {custom_result_dict}"
    )