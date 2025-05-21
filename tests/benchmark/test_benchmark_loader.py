import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import torch
from datasets import Dataset, IterableDataset # For spec and type comparison
import re

from src.benchmark.benchmark_loader import BenchmarkRunner, DatasetFactory, ModelLoaderFactory, Evaluator
from src.config_models import (
    BenchmarkConfig, GeneralConfig, ModelConfig, TaskConfig,
    DatasetConfig, MetricConfig, ReportingConfig, EvaluationConfig,
    AdvancedConfig, ModelParamsConfig
)

# --- Fixtures ---
@pytest.fixture
def minimal_general_config():
    return GeneralConfig(output_dir=Path("test_benchmark_outputs"), experiment_name="test_exp")

@pytest.fixture
def minimal_reporting_config():
    return ReportingConfig(format="json", output_dir=Path("test_reports"))

@pytest.fixture
def minimal_advanced_config():
    return AdvancedConfig(batch_size=2, max_new_tokens=10)

@pytest.fixture
def minimal_evaluation_config():
    return EvaluationConfig(log_interval=1)

@pytest.fixture
def minimal_model_params_config():
    return ModelParamsConfig()

@pytest.fixture
def dummy_model_config_payload():
    return {"name": "dummy-model", "framework": "huggingface", "checkpoint": "dummy/model-checkpoint"}

@pytest.fixture
def dummy_model_config(dummy_model_config_payload):
    return ModelConfig(**dummy_model_config_payload)

@pytest.fixture
def dummy_dataset_config_payload():
    return {"name": "dummy-dataset", "source_type": "hf_hub", "split": "train", "max_samples": 10}

@pytest.fixture
def dummy_dataset_config(dummy_dataset_config_payload):
    return DatasetConfig(**dummy_dataset_config_payload)

@pytest.fixture
def dummy_metric_config_payload():
    return {"name": "accuracy"}

@pytest.fixture
def dummy_metric_config(dummy_metric_config_payload):
    return MetricConfig(**dummy_metric_config_payload)

@pytest.fixture
def dummy_task_config_obj(dummy_dataset_config, dummy_metric_config):
    return TaskConfig(
        name="dummy-task", type="classification", datasets=[dummy_dataset_config],
        evaluation_metrics=[dummy_metric_config],
        handler_options={"prompt_builder_type": "default"}
    )

@pytest.fixture
def basic_benchmark_config(
    minimal_general_config, dummy_task_config_obj, dummy_model_config,
    minimal_model_params_config, minimal_evaluation_config,
    minimal_reporting_config, minimal_advanced_config
):
    return BenchmarkConfig(
        general=minimal_general_config, tasks=[dummy_task_config_obj], models=[dummy_model_config],
        model_parameters=minimal_model_params_config, evaluation=minimal_evaluation_config,
        reporting=minimal_reporting_config, advanced=minimal_advanced_config
    )

@pytest.fixture
def dummy_model_config_alt(dummy_model_config_payload):
    # Alternate model config for multi-model tests
    alt_model_payload = dummy_model_config_payload.copy()
    alt_model_payload["name"] = "alt-dummy-model"
    alt_model_payload["checkpoint"] = "dummy/alt-model-checkpoint"
    return ModelConfig(**alt_model_payload)

@pytest.fixture
def alt_dataset_config_payload():
    return {"name": "alt-dummy-dataset", "source_type": "hf_hub", "split": "test", "max_samples": 5}

@pytest.fixture
def alt_dataset_config(alt_dataset_config_payload):
    return DatasetConfig(**alt_dataset_config_payload)

@pytest.fixture
def alt_metric_config_payload():
    return {"name": "f1_score"}

@pytest.fixture
def alt_metric_config(alt_metric_config_payload):
    return MetricConfig(**alt_metric_config_payload)
# --- End of New Fixtures ---

@pytest.fixture
def dummy_task_config_obj_alt(alt_dataset_config, alt_metric_config): # Assicurati che le dipendenze siano corrette
    # Alternate task config for multi-task tests
    return TaskConfig(
        name="alt-dummy-task", 
        type="classification", 
        datasets=[alt_dataset_config], # Usa la nuova fixture
        evaluation_metrics=[alt_metric_config], # Usa la nuova fixture
        handler_options={"prompt_builder_type": "default"}
    )

# --- Test Files (Ensure this is the SUT code provided by user for BenchmarkRunner) ---
# The benchmark_loader.py file provided by the user should be used as the SUT.

# --- BenchmarkRunner ---

@patch('src.benchmark.benchmark_loader.setup_logger')
def test_benchmark_runner_initialization(mock_setup_logger_sut, basic_benchmark_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    with patch('torch.cuda.is_available', return_value=False): # Ensure predictable device
        runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance # Explicitly set for clarity in test assertions
    assert runner.config == basic_benchmark_config
    found_runner_logger_call = any(
        call_obj.args and call_obj.args[0] == BenchmarkRunner.__name__
        for call_obj in mock_setup_logger_sut.call_args_list
    )
    assert found_runner_logger_call, f"setup_logger calls: {mock_setup_logger_sut.call_args_list}"

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('torch.cuda.is_available', return_value=True)
def test_benchmark_runner_determine_device_cuda(mock_torch_cuda_is_available, mock_setup_logger_sut, basic_benchmark_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    assert runner.device == "cuda"
    mock_logger_instance.info.assert_any_call("CUDA available. Using GPU.")

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('torch.cuda.is_available', return_value=False)
def test_benchmark_runner_determine_device_cpu(mock_torch_cuda_is_available, mock_setup_logger_sut, basic_benchmark_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    assert runner.device == "cpu"
    mock_logger_instance.info.assert_any_call("No GPU detected. Using CPU.")

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.save_results')
def test_benchmark_runner_save_results_called(mock_save_results_fn, mock_setup_logger_sut, basic_benchmark_config, tmp_path):
    basic_benchmark_config.general.output_dir = tmp_path
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    runner.results = {basic_benchmark_config.models[0].name: {"dummy-task": {"accuracy": 0.9}}}
    runner._save_results()
    mock_save_results_fn.assert_called_once_with(
        results=runner.results, output_dir=str(tmp_path),
        format=basic_benchmark_config.reporting.format
    )

@patch('src.benchmark.benchmark_loader.setup_logger')
def test_benchmark_runner_save_results_no_results(mock_setup_logger_sut, basic_benchmark_config, tmp_path):
    basic_benchmark_config.general.output_dir = tmp_path
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    runner.results = {}
    runner._save_results()
    runner.logger.warning.assert_called_with("No benchmark results were generated to save.")

@patch('gc.collect')
@patch('torch.cuda.empty_cache')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._helper_disk_cache_cleanup')
@patch('torch.cuda.is_available', return_value=True)
@patch('src.benchmark.benchmark_loader.setup_logger')
def test_benchmark_runner_cleanup_model_resources(
    mock_setup_logger_sut, mock_cuda_is_available, mock_disk_cleanup_helper,
    mock_empty_cache, mock_gc_collect, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    mock_model = MagicMock(name="MockModel")
    mock_tokenizer = MagicMock(name="MockTokenizer")
    model_cfg = dummy_model_config
    runner._cleanup_model_resources(mock_model, mock_tokenizer, model_cfg)
    mock_gc_collect.assert_called_once()
    mock_empty_cache.assert_called_once()
    mock_disk_cleanup_helper.assert_not_called()
    mock_logger_instance.info.assert_any_call(f"Cleaning up memory resources for model '{model_cfg.checkpoint}'...")
    mock_logger_instance.info.assert_any_call(f"Memory resources cleaned up for model '{model_cfg.checkpoint}'.")

@patch('gc.collect')
@patch('torch.cuda.empty_cache')
@patch('torch.cuda.is_available', return_value=False)
@patch('src.benchmark.benchmark_loader.setup_logger')
def test_benchmark_runner_cleanup_model_resources_no_cuda(
    mock_setup_logger_sut, mock_cuda_is_available, mock_empty_cache, mock_gc_collect,
    basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    model_cfg_for_cleanup = dummy_model_config
    runner._cleanup_model_resources(mock_model, mock_tokenizer, model_cfg_for_cleanup)
    mock_gc_collect.assert_called_once()
    mock_empty_cache.assert_not_called()
    mock_logger_instance.info.assert_any_call(f"Cleaning up memory resources for model '{model_cfg_for_cleanup.checkpoint}'...")

# --- Corrected Tests for _load_single_dataset ---

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.DatasetFactory')
def test_benchmark_runner_load_single_dataset_basic_flow_and_max_samples_in_config(
    mock_dataset_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_task_config_obj
):
    """
    Tests that _load_single_dataset calls the factory with correct config (incl. max_samples)
    and calls loader.load(). It does not assert .select() or .take() as that's loader's job.
    """
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    # 1. Test with max_samples set in the task_config
    task_cfg_with_max_samples = dummy_task_config_obj.model_copy(deep=True)
    task_cfg_with_max_samples.datasets[0].max_samples = 5 # Specific value for this test part

    mock_loader_instance_1 = MagicMock()
    mock_loaded_dataset_1 = MagicMock(spec=Dataset) # This is what loader.load() returns
    mock_loader_instance_1.load.return_value = mock_loaded_dataset_1
    mock_loader_instance_1.streaming = False # Example attribute
    mock_dataset_factory.from_config.return_value = mock_loader_instance_1

    # SUT Call 1
    dataset_info_1 = runner._load_single_dataset(task_cfg=task_cfg_with_max_samples)

    assert dataset_info_1 is not None
    assert dataset_info_1["dataset"] is mock_loaded_dataset_1
    assert dataset_info_1["streaming"] is False # Based on mock_loader_instance_1.streaming
    assert dataset_info_1["task_config"] == task_cfg_with_max_samples

    expected_ds_config_dict_1 = task_cfg_with_max_samples.datasets[0].model_dump(exclude_none=True)
    assert expected_ds_config_dict_1['max_samples'] == 5 # Verify max_samples was in the config passed to factory
    mock_dataset_factory.from_config.assert_called_with(expected_ds_config_dict_1) # Check the whole config
    mock_loader_instance_1.load.assert_called_once_with(task_type=task_cfg_with_max_samples.type)
    mock_logger_instance.debug.assert_any_call(f"Dataset '{task_cfg_with_max_samples.datasets[0].name}' loaded for task '{task_cfg_with_max_samples.name}'.")

    # Reset factory mock for next call if it's the same mock object for all tests in the class
    mock_dataset_factory.reset_mock()
    mock_loader_instance_1.reset_mock() # Reset the loader mock as well

    # 2. Test with a different max_samples or default from fixture (which is 10)
    #    This ensures the test isn't just passing due to one specific value.
    #    Using original dummy_task_config_obj which has max_samples = 10
    mock_loader_instance_2 = MagicMock()
    mock_loaded_dataset_2 = MagicMock(spec=IterableDataset)
    mock_loader_instance_2.load.return_value = mock_loaded_dataset_2
    mock_loader_instance_2.streaming = True
    mock_dataset_factory.from_config.return_value = mock_loader_instance_2
    
    # SUT Call 2
    dataset_info_2 = runner._load_single_dataset(task_cfg=dummy_task_config_obj) # Uses original max_samples = 10

    assert dataset_info_2 is not None
    assert dataset_info_2["dataset"] is mock_loaded_dataset_2
    assert dataset_info_2["streaming"] is True

    expected_ds_config_dict_2 = dummy_task_config_obj.datasets[0].model_dump(exclude_none=True)
    assert expected_ds_config_dict_2['max_samples'] == 10 # From original fixture
    mock_dataset_factory.from_config.assert_called_with(expected_ds_config_dict_2)
    mock_loader_instance_2.load.assert_called_once_with(task_type=dummy_task_config_obj.type)

# The previous separate tests for map-style and iterable-style max_samples are now effectively
# covered by the single test above, as BenchmarkRunner._load_single_dataset does not differentiate
# its own direct actions based on map/iterable for max_samples. It passes the config to the factory.
# The specific .select() or .take() behavior should be tested in ConcreteDatasetLoader tests.

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.DatasetFactory')
def test_benchmark_runner_load_single_dataset_factory_fails(
    mock_dataset_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    mock_dataset_factory.from_config.side_effect = ValueError("Factory error")
    dataset_info = runner._load_single_dataset(task_cfg=dummy_task_config_obj)
    assert dataset_info is None
    mock_logger_instance.error.assert_called_once()
    args, _ = mock_logger_instance.error.call_args
    assert f"Failed to load dataset for task '{dummy_task_config_obj.name}': Factory error" in args[0]


# --- Tests for _load_model_and_tokenizer (ensure logger is mocked correctly) ---
@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.ModelLoaderFactory')
def test_benchmark_runner_load_model_and_tokenizer_success_no_offload_no_quant(
    mock_model_loader_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    with patch('torch.cuda.is_available', return_value=True):
        runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_model_instance = MagicMock(spec_set=['to', 'hf_device_map', 'device', 'name_or_path', 'parameters', 'config'])
    mock_model_instance.hf_device_map = None
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.device = runner.device
    mock_model_instance.name_or_path = dummy_model_config.checkpoint
    mock_model_instance.config = MagicMock(is_encoder_decoder=False, pad_token_id=None)

    mock_tokenizer_instance = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = (mock_model_instance, mock_tokenizer_instance)
    mock_model_loader_factory.get_model_loader.return_value = mock_loader_instance
    current_dummy_model_config = dummy_model_config.model_copy(deep=True)
    current_dummy_model_config.quantization = None

    with patch('src.benchmark.benchmark_loader.isinstance') as mock_isinstance_dataparallel:
        mock_isinstance_dataparallel.side_effect = lambda obj, classinfo: False if classinfo is torch.nn.DataParallel else __builtins__.isinstance(obj, classinfo)
        model_tuple = runner._load_model_and_tokenizer(model_cfg=current_dummy_model_config)

    assert model_tuple is not None, f"Model tuple was None. Log calls: {mock_logger_instance.mock_calls}"
    model, tokenizer = model_tuple
    assert model is mock_model_instance
    assert tokenizer is mock_tokenizer_instance
    model_specific_params = current_dummy_model_config.model_dump(exclude_none=True)
    global_model_params = basic_benchmark_config.model_parameters.model_dump(exclude_none=True)
    mock_model_loader_factory.get_model_loader.assert_called_once_with(
        model_name=current_dummy_model_config.checkpoint, framework=current_dummy_model_config.framework,
        quantization=None, model_specific_config_params=model_specific_params,
        global_model_creation_params=global_model_params
    )
    mock_loader_instance.load.assert_called_once_with(quantization=None)
    if runner.device == "cuda":
        if not (hasattr(mock_model_instance, 'hf_device_map') and mock_model_instance.hf_device_map is not None):
            if hasattr(mock_model_instance, 'to') and callable(mock_model_instance.to):
                mock_model_instance.to.assert_called_once_with(runner.device)
    mock_logger_instance.info.assert_any_call(f"Moving model '{current_dummy_model_config.name}' to device '{runner.device}'.")
    mock_logger_instance.info.assert_any_call(f"Model '{current_dummy_model_config.name}' loaded. Effective device(s): {runner.device}.")

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.ModelLoaderFactory')
def test_benchmark_runner_load_model_and_tokenizer_quantized_model(
    mock_model_loader_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    # Use spec_set, omitting 'device' to test the hf_device_map logging path
    mock_model_instance = MagicMock(spec_set=['hf_device_map', 'to', 'name_or_path', 'parameters', 'config'])
    mock_model_instance.hf_device_map = {"": 0}
    mock_model_instance.to = MagicMock()
    mock_model_instance.name_or_path = dummy_model_config.checkpoint
    mock_model_instance.config = MagicMock(is_encoder_decoder=False, pad_token_id=None)

    mock_tokenizer_instance = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = (mock_model_instance, mock_tokenizer_instance)
    mock_model_loader_factory.get_model_loader.return_value = mock_loader_instance
    current_dummy_model_config = dummy_model_config.model_copy(deep=True)
    current_dummy_model_config.quantization = "4bit"

    model_tuple = runner._load_model_and_tokenizer(model_cfg=current_dummy_model_config)

    assert model_tuple is not None
    mock_loader_instance.load.assert_called_once_with(quantization="4bit")
    mock_model_instance.to.assert_not_called()
    mock_logger_instance.info.assert_any_call(
        f"Model '{current_dummy_model_config.name}' loaded. Effective device(s): distributed via hf_device_map: {{'': 0}}."
    )

# Include other _factory_fails and _loader_load_fails tests for models here,
# ensuring they also mock setup_logger.
@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.ModelLoaderFactory')
def test_benchmark_runner_load_model_and_tokenizer_factory_fails(
    mock_model_loader_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_model_loader_factory.get_model_loader.side_effect = ValueError("Factory get_model_loader error")

    model_tuple = runner._load_model_and_tokenizer(model_cfg=dummy_model_config)
    assert model_tuple is None
    mock_logger_instance.error.assert_called_once()
    args, kwargs = mock_logger_instance.error.call_args
    assert f"Failed to load model '{dummy_model_config.name}': Factory get_model_loader error" in args[0]


# --- Tests for main run() method (Example Structure) ---
# These will require more extensive mocking of internal methods of BenchmarkRunner

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_model_and_tokenizer')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_single_dataset')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._run_task_evaluation')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._cleanup_model_resources')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._save_results')
def test_run_method_orchestration_single_model_task(
    mock_save_results, mock_cleanup_resources, mock_run_task_evaluation,
    mock_load_single_dataset, mock_load_model_tokenizer, mock_setup_logger_sut,
    basic_benchmark_config, dummy_model_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    # Setup return values for mocked methods
    mock_model = MagicMock(name="MockLoadedModel")
    mock_tokenizer = MagicMock(name="MockLoadedTokenizer")
    mock_load_model_tokenizer.return_value = (mock_model, mock_tokenizer)

    mock_dataset_obj = MagicMock(spec=Dataset, name="MockLoadedDataset")
    mock_load_single_dataset.return_value = {"dataset": mock_dataset_obj, "name": "dummy-dataset", "task_config": dummy_task_config_obj, "streaming": False}
    
    mock_run_task_evaluation.return_value = {"accuracy": 0.95} # Simulate evaluation results

    # Call the run method
    results = runner.run()

    # Assertions
    mock_load_model_tokenizer.assert_called_once_with(dummy_model_config)
    mock_load_single_dataset.assert_called_once_with(dummy_task_config_obj)
    mock_run_task_evaluation.assert_called_once_with(mock_model, mock_tokenizer, dummy_task_config_obj, mock_load_single_dataset.return_value)
    mock_cleanup_resources.assert_called_once_with(mock_model, mock_tokenizer, dummy_model_config)
    
    # _save_results is called after each model and at the very end
    assert mock_save_results.call_count == 2 

    assert dummy_model_config.name in results
    assert dummy_task_config_obj.name in results[dummy_model_config.name]
    assert results[dummy_model_config.name][dummy_task_config_obj.name] == {"accuracy": 0.95}

    mock_logger_instance.info.assert_any_call("All models processed. Saving final benchmark results...")


@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._save_results')
def test_run_method_no_models(mock_save_results, mock_setup_logger_sut, basic_benchmark_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    
    config_no_models = basic_benchmark_config.model_copy(deep=True)
    config_no_models.models = [] # No models defined
    
    runner = BenchmarkRunner(config=config_no_models)
    runner.logger = mock_logger_instance
    
    results = runner.run()
    
    assert results == {}
    runner.logger.error.assert_called_with("No models defined in the configuration. Aborting.")
    mock_save_results.assert_not_called() # Should not save if no models run

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_model_and_tokenizer')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._cleanup_model_resources')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._save_results')
def test_run_method_model_load_fails(
    mock_save_results, mock_cleanup_resources, mock_load_model_tokenizer, 
    mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    
    runner = BenchmarkRunner(config=basic_benchmark_config) # Config has one model
    runner.logger = mock_logger_instance
    
    mock_load_model_tokenizer.return_value = None # Simulate model loading failure
    
    results = runner.run()
    
    mock_load_model_tokenizer.assert_called_once_with(dummy_model_config)
    assert dummy_model_config.name in results
    assert results[dummy_model_config.name]["error"] == "Model loading failed"
    mock_cleanup_resources.assert_not_called() # Cleanup should not be called if model load failed
    
    # _save_results is called once after the failed model, and once at the very end
    assert mock_save_results.call_count == 2
    runner.logger.info.assert_any_call("All models processed. Saving final benchmark results...")

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_model_and_tokenizer')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._cleanup_model_resources')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._save_results')
def test_run_method_no_tasks(
    mock_save_results, mock_cleanup_resources, mock_load_model_tokenizer, 
    mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    
    config_no_tasks = basic_benchmark_config.model_copy(deep=True)
    config_no_tasks.tasks = [] # No tasks defined
    
    runner = BenchmarkRunner(config=config_no_tasks)
    runner.logger = mock_logger_instance
    
    mock_model_instance = MagicMock(name="MockLoadedModel")
    mock_tokenizer_instance = MagicMock(name="MockLoadedTokenizer")
    mock_load_model_tokenizer.return_value = (mock_model_instance, mock_tokenizer_instance)
    
    results = runner.run()
    
    mock_load_model_tokenizer.assert_called_once_with(dummy_model_config)
    assert dummy_model_config.name in results
    assert results[dummy_model_config.name] == {} # No tasks, so empty dict for the model
    mock_cleanup_resources.assert_called_once_with(mock_model_instance, mock_tokenizer_instance, dummy_model_config)
    assert mock_save_results.call_count == 2 # Called after model and at the end

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_model_and_tokenizer')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_single_dataset')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._run_task_evaluation')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._cleanup_model_resources')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._save_results')
def test_run_method_dataset_load_fails_for_task(
    mock_save_results, mock_cleanup_resources, mock_run_task_evaluation,
    mock_load_single_dataset, mock_load_model_tokenizer, mock_setup_logger_sut,
    basic_benchmark_config, dummy_model_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    
    runner = BenchmarkRunner(config=basic_benchmark_config) # Has one model, one task
    runner.logger = mock_logger_instance

    mock_model_instance = MagicMock(name="MockLoadedModel")
    mock_tokenizer_instance = MagicMock(name="MockLoadedTokenizer")
    mock_load_model_tokenizer.return_value = (mock_model_instance, mock_tokenizer_instance)
    
    mock_load_single_dataset.return_value = None # Simulate dataset loading failure
    
    results = runner.run()
    
    mock_load_model_tokenizer.assert_called_once_with(dummy_model_config)
    mock_load_single_dataset.assert_called_once_with(dummy_task_config_obj)
    
    assert dummy_model_config.name in results
    assert dummy_task_config_obj.name in results[dummy_model_config.name]
    assert results[dummy_model_config.name][dummy_task_config_obj.name]["error"] == "Dataset loading failed"
    
    mock_run_task_evaluation.assert_not_called() # Task evaluation should not be called
    mock_cleanup_resources.assert_called_once_with(mock_model_instance, mock_tokenizer_instance, dummy_model_config)
    assert mock_save_results.call_count == 2

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_model_and_tokenizer')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_single_dataset')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._run_task_evaluation')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._cleanup_model_resources')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._save_results')
def test_run_method_task_evaluation_fails(
    mock_save_results, mock_cleanup_resources, mock_run_task_evaluation,
    mock_load_single_dataset, mock_load_model_tokenizer, mock_setup_logger_sut,
    basic_benchmark_config, dummy_model_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_model_instance = MagicMock(name="MockLoadedModel")
    mock_tokenizer_instance = MagicMock(name="MockLoadedTokenizer")
    mock_load_model_tokenizer.return_value = (mock_model_instance, mock_tokenizer_instance)
    
    mock_dataset_obj = MagicMock(spec=Dataset, name="MockLoadedDataset")
    mock_dataset_info = {"dataset": mock_dataset_obj, "name": "dummy-dataset", "task_config": dummy_task_config_obj, "streaming": False}
    mock_load_single_dataset.return_value = mock_dataset_info
    
    mock_run_task_evaluation.side_effect = Exception("Task evaluation error") # Simulate failure in task evaluation
    
    results = runner.run()
    
    mock_load_model_tokenizer.assert_called_once_with(dummy_model_config)
    mock_load_single_dataset.assert_called_once_with(dummy_task_config_obj)
    mock_run_task_evaluation.assert_called_once_with(mock_model_instance, mock_tokenizer_instance, dummy_task_config_obj, mock_dataset_info)
    
    assert dummy_model_config.name in results
    assert dummy_task_config_obj.name in results[dummy_model_config.name]
    assert "Task execution/evaluation failed: Task evaluation error" in results[dummy_model_config.name][dummy_task_config_obj.name]["error"]
    
    mock_cleanup_resources.assert_called_once_with(mock_model_instance, mock_tokenizer_instance, dummy_model_config)
    assert mock_save_results.call_count == 2

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_model_and_tokenizer')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._load_single_dataset')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._run_task_evaluation')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._cleanup_model_resources')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._save_results')
def test_run_method_multiple_models_and_tasks(
    mock_save_results, mock_cleanup_resources, mock_run_task_evaluation,
    mock_load_single_dataset, mock_load_model_tokenizer, mock_setup_logger_sut,
    minimal_general_config, minimal_model_params_config, minimal_evaluation_config,
    minimal_reporting_config, minimal_advanced_config,
    dummy_model_config, dummy_model_config_alt, 
    dummy_task_config_obj, dummy_task_config_obj_alt
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance

    multi_config = BenchmarkConfig(
        general=minimal_general_config,
        tasks=[dummy_task_config_obj, dummy_task_config_obj_alt],
        models=[dummy_model_config, dummy_model_config_alt],
        model_parameters=minimal_model_params_config,
        evaluation=minimal_evaluation_config,
        reporting=minimal_reporting_config,
        advanced=minimal_advanced_config
    )
    runner = BenchmarkRunner(config=multi_config)
    runner.logger = mock_logger_instance

    # Mock loaded models and tokenizers
    mock_model1, mock_tokenizer1 = MagicMock(name="Model1"), MagicMock(name="Tokenizer1")
    mock_model2, mock_tokenizer2 = MagicMock(name="Model2"), MagicMock(name="Tokenizer2")
    mock_load_model_tokenizer.side_effect = [
        (mock_model1, mock_tokenizer1),
        (mock_model2, mock_tokenizer2)
    ]

    # Mock loaded datasets
    mock_dataset1 = MagicMock(spec=Dataset, name="Dataset1")
    mock_dataset2 = MagicMock(spec=Dataset, name="Dataset2")
    mock_dataset_info1 = {"dataset": mock_dataset1, "name": "ds1", "task_config": dummy_task_config_obj, "streaming": False}
    mock_dataset_info2 = {"dataset": mock_dataset2, "name": "ds2", "task_config": dummy_task_config_obj_alt, "streaming": False}
    
    # _load_single_dataset will be called for each task within each model's loop
    # For Model1: Task1, Task2. For Model2: Task1, Task2
    mock_load_single_dataset.side_effect = [
        mock_dataset_info1, mock_dataset_info2, # For Model 1
        mock_dataset_info1, mock_dataset_info2  # For Model 2
    ]
    
    # Mock evaluation results
    # Model1-Task1, Model1-Task2, Model2-Task1, Model2-Task2
    mock_run_task_evaluation.side_effect = [
        {"metric1": 0.9}, {"metric_alt": 0.8},
        {"metric1": 0.95}, {"metric_alt": 0.85}
    ]

    results = runner.run()

    # Assertions for model loading
    assert mock_load_model_tokenizer.call_count == 2
    mock_load_model_tokenizer.assert_any_call(dummy_model_config)
    mock_load_model_tokenizer.assert_any_call(dummy_model_config_alt)

    # Assertions for dataset loading
    assert mock_load_single_dataset.call_count == 4 # 2 models * 2 tasks
    mock_load_single_dataset.assert_any_call(dummy_task_config_obj)    # Called twice
    mock_load_single_dataset.assert_any_call(dummy_task_config_obj_alt) # Called twice
    
    # Assertions for task evaluation
    assert mock_run_task_evaluation.call_count == 4
    mock_run_task_evaluation.assert_any_call(mock_model1, mock_tokenizer1, dummy_task_config_obj, mock_dataset_info1)
    mock_run_task_evaluation.assert_any_call(mock_model1, mock_tokenizer1, dummy_task_config_obj_alt, mock_dataset_info2)
    mock_run_task_evaluation.assert_any_call(mock_model2, mock_tokenizer2, dummy_task_config_obj, mock_dataset_info1) # Dataset info re-mocked for model2
    mock_run_task_evaluation.assert_any_call(mock_model2, mock_tokenizer2, dummy_task_config_obj_alt, mock_dataset_info2)


    # Assertions for cleanup
    assert mock_cleanup_resources.call_count == 2
    mock_cleanup_resources.assert_any_call(mock_model1, mock_tokenizer1, dummy_model_config)
    mock_cleanup_resources.assert_any_call(mock_model2, mock_tokenizer2, dummy_model_config_alt)

    # Assertions for results saving
    # Called after each model (2 times) and once at the very end (1 time)
    assert mock_save_results.call_count == 3 

    # Check final results structure
    assert dummy_model_config.name in results
    assert dummy_task_config_obj.name in results[dummy_model_config.name]
    assert results[dummy_model_config.name][dummy_task_config_obj.name] == {"metric1": 0.9}
    assert dummy_task_config_obj_alt.name in results[dummy_model_config.name]
    assert results[dummy_model_config.name][dummy_task_config_obj_alt.name] == {"metric_alt": 0.8}

    assert dummy_model_config_alt.name in results
    assert dummy_task_config_obj.name in results[dummy_model_config_alt.name]
    assert results[dummy_model_config_alt.name][dummy_task_config_obj.name] == {"metric1": 0.95}
    assert dummy_task_config_obj_alt.name in results[dummy_model_config_alt.name]
    assert results[dummy_model_config_alt.name][dummy_task_config_obj_alt.name] == {"metric_alt": 0.85}

    runner.logger.info.assert_any_call(f"Saving intermediate results after processing model '{dummy_model_config.name}'...")
    runner.logger.info.assert_any_call(f"Saving intermediate results after processing model '{dummy_model_config_alt.name}'...")
    runner.logger.info.assert_any_call("All models processed. Saving final benchmark results...")


# --- Tests for _helper_disk_cache_cleanup ---
@patch('src.benchmark.benchmark_loader.setup_logger')
def test_helper_disk_cache_cleanup_disabled_by_default(mock_setup_logger_sut, basic_benchmark_config, dummy_model_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    model_cfg_with_cleanup = dummy_model_config.model_copy(deep=True)
    model_cfg_with_cleanup.cleanup_model_cache_after_run = True # Request cleanup
    # Checkpoint is "dummy/model-checkpoint"

    with pytest.raises(RuntimeError, match="Disk cleanup is disabled by default for safety."):
        runner._helper_disk_cache_cleanup(model_cfg_with_cleanup)

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch.dict('os.environ', {"HF_HOME": "/test/hf_home_env"}, clear=True)
@patch('pathlib.Path.is_dir')
@patch('pathlib.Path.resolve')
# Mock the actual shutil.rmtree if we were testing the deletion itself
# @patch('shutil.rmtree') 
def test_helper_disk_cache_cleanup_safety_checks_and_flow_if_enabled(
    mock_path_resolve, mock_path_is_dir, 
    mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
    # mock_shutil_rmtree # If testing actual deletion
):
    # THIS TEST IS DESIGNED FOR THE SCENARIO WHERE THE RuntimeError
    # IN _helper_disk_cache_cleanup IS COMMENTED OUT.
    # As it stands, it will fail because the RuntimeError is raised first.
    # To make this test pass, you'd need to modify the SUT.
    pytest.skip("Skipping test as it requires modification of SUT (_helper_disk_cache_cleanup) to disable RuntimeError.")

    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    model_cfg_with_cleanup = dummy_model_config.model_copy(deep=True)
    model_cfg_with_cleanup.cleanup_model_cache_after_run = True
    model_cfg_with_cleanup.checkpoint = "org/test-model" # HF hub style

    # --- Mock Path behaviors ---
    # Simulate HF_HOME and hub structure
    # Path("/test/hf_home_env").resolve() -> "/test/hf_home_env"
    # Path("/test/hf_home_env/hub").resolve() -> "/test/hf_home_env/hub"
    # Path("/test/hf_home_env/hub/models--org--test-model").resolve() -> "/test/hf_home_env/hub/models--org--test-model"
    
    def path_resolve_side_effect(*args, **kwargs):
        # Based on the path instance calling resolve
        if str(args[0]) == "/test/hf_home_env":
            return Path("/test/hf_home_env")
        elif str(args[0]) == "/test/hf_home_env/hub":
            return Path("/test/hf_home_env/hub")
        elif str(args[0]) == "/test/hf_home_env/hub/models--org--test-model":
            return Path("/test/hf_home_env/hub/models--org--test-model")
        return args[0] # Default pass-through

    mock_path_resolve.side_effect = path_resolve_side_effect
    
    # Path.is_dir() behavior
    def path_is_dir_side_effect(path_instance):
        if str(path_instance) == "/test/hf_home_env/hub":
            return True # Hub cache exists
        if str(path_instance) == "/test/hf_home_env/hub/models--org--test-model":
            return True # Model cache dir exists
        return False
    mock_path_is_dir.side_effect = path_is_dir_side_effect
    
    # --- SUT Call (assuming RuntimeError is commented out in SUT) ---
    # runner._helper_disk_cache_cleanup(model_cfg_with_cleanup)

    # --- Assertions (if RuntimeError was commented out) ---
    # mock_shutil_rmtree.assert_called_once_with(Path("/test/hf_home_env/hub/models--org--test-model"))
    # mock_logger_instance.info.assert_any_call("All safety checks passed. Proceeding to delete model cache directory: /test/hf_home_env/hub/models--org--test-model")
    # mock_logger_instance.info.assert_any_call("Successfully deleted disk cache for model 'org/test-model' from '/test/hf_home_env/hub/models--org--test-model'.")


@patch('src.benchmark.benchmark_loader.setup_logger')
def test_helper_disk_cache_cleanup_no_checkpoint(mock_setup_logger_sut, basic_benchmark_config, dummy_model_config):
    # THIS TEST IS DESIGNED FOR THE SCENARIO WHERE THE RuntimeError
    # IN _helper_disk_cache_cleanup IS COMMENTED OUT.
    pytest.skip("Skipping test as it requires modification of SUT (_helper_disk_cache_cleanup) to disable RuntimeError.")

    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    model_cfg_no_checkpoint = dummy_model_config.model_copy(deep=True)
    model_cfg_no_checkpoint.cleanup_model_cache_after_run = True
    model_cfg_no_checkpoint.checkpoint = None # No checkpoint

    # --- SUT Call (assuming RuntimeError is commented out in SUT) ---
    # runner._helper_disk_cache_cleanup(model_cfg_no_checkpoint)
    
    # --- Assertions (if RuntimeError was commented out) ---
    # runner.logger.warning.assert_called_with(
    #     f"Cleanup model cache requested for '{model_cfg_no_checkpoint.name}', but 'checkpoint' (model ID for cache) is not defined. Cannot determine cache path."
    # )


# --- Additional tests for _load_single_dataset ---

@patch('src.benchmark.benchmark_loader.setup_logger')
def test_load_single_dataset_no_datasets_in_task_config(mock_setup_logger_sut, basic_benchmark_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    task_cfg_no_datasets = basic_benchmark_config.tasks[0].model_copy(deep=True)
    task_cfg_no_datasets.datasets = [] # Empty list of datasets

    dataset_info = runner._load_single_dataset(task_cfg=task_cfg_no_datasets)

    assert dataset_info is None
    mock_logger_instance.warning.assert_called_with(
        f"Task '{task_cfg_no_datasets.name}' has no datasets defined. Skipping dataset loading for this task."
    )

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.DatasetFactory')
def test_load_single_dataset_loader_load_returns_none(
    mock_dataset_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = None # Simulate loader.load() returning None
    mock_loader_instance.streaming = False 
    mock_dataset_factory.from_config.return_value = mock_loader_instance

    dataset_info = runner._load_single_dataset(task_cfg=dummy_task_config_obj) # SUT call
    
    # Assertions
    assert dataset_info is not None 
    assert dataset_info["dataset"] is None 
    assert dataset_info["streaming"] is False
    assert dataset_info["task_config"] == dummy_task_config_obj
    
    # Check for the log message indicating dataset loading
    mock_logger_instance.error.assert_not_called() 
    
    # Check for the log message indicating dataset loading
    mock_logger_instance.debug.assert_any_call(
        f"Dataset '{dummy_task_config_obj.datasets[0].name}' loaded for task '{dummy_task_config_obj.name}'."
    )

# --- Additional tests for _load_model_and_tokenizer ---

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.ModelLoaderFactory')
def test_load_model_and_tokenizer_model_is_dataparallel(
    mock_model_loader_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance
    runner.device = "cuda" # Ensure we test the .to(device) path

    # Simulate a model that is an instance of DataParallel
    mock_model_dataparallel = MagicMock(spec=torch.nn.DataParallel)
    # Add attributes that would be checked before .to() call
    mock_model_dataparallel.hf_device_map = None 
    mock_model_dataparallel.name_or_path = dummy_model_config.checkpoint
    # spec_set for config if its attributes are accessed
    mock_model_dataparallel.config = MagicMock(is_encoder_decoder=False, pad_token_id=None) 


    mock_tokenizer_instance = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = (mock_model_dataparallel, mock_tokenizer_instance)
    mock_model_loader_factory.get_model_loader.return_value = mock_loader_instance

    model_cfg_no_quant = dummy_model_config.model_copy(deep=True)
    model_cfg_no_quant.quantization = None

    # Patch isinstance to make it return True for torch.nn.DataParallel
    with patch('src.benchmark.benchmark_loader.isinstance', side_effect=lambda obj, classinfo: True if classinfo is torch.nn.DataParallel else __builtins__.isinstance(obj, classinfo)):
        model_tuple = runner._load_model_and_tokenizer(model_cfg=model_cfg_no_quant)

    assert model_tuple is not None
    loaded_model, _ = model_tuple
    assert loaded_model is mock_model_dataparallel
    
    # Ensure model.to(device) was NOT called because it's DataParallel
    mock_model_dataparallel.to.assert_not_called() 
    
    # Check for the log message indicating device placement
    # For DataParallel, it might just log the general device or a message about DataParallel handling.
    # The current SUT doesn't have a specific log for DataParallel, so it might fall into the "Handled by loader"
    # or just the default device if no explicit .to call is made and no device_map.
    # We expect it *not* to log the "Moving model..." message.
    log_calls_str = "".join([str(call_args) for call_args in mock_logger_instance.info.call_args_list])
    assert f"Moving model '{model_cfg_no_quant.name}' to device '{runner.device}'" not in log_calls_str
    mock_logger_instance.info.assert_any_call(f"Model '{model_cfg_no_quant.name}' loaded. Effective device(s): {runner.device}.")


@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.ModelLoaderFactory')
def test_load_model_and_tokenizer_loader_load_fails_non_value_error(
    mock_model_loader_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_model_config
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.side_effect = TypeError("Simulated TypeError from loader.load()")
    mock_model_loader_factory.get_model_loader.return_value = mock_loader_instance

    model_tuple = runner._load_model_and_tokenizer(model_cfg=dummy_model_config)

    assert model_tuple is None
    mock_logger_instance.error.assert_called_once()
    args, _ = mock_logger_instance.error.call_args
    assert f"Failed to load model '{dummy_model_config.name}': Simulated TypeError from loader.load()" in args[0]

# --- Tests for _process_task_batches ---

@patch('src.benchmark.benchmark_loader.setup_logger')
def test_process_task_batches_invalid_log_interval(mock_setup_logger_sut, basic_benchmark_config, dummy_task_config_obj):
    mock_runner_logger = MagicMock()
    mock_handler_logger = MagicMock() # Separate logger for handler if needed
    mock_evaluator_logger = MagicMock()

    # Patch setup_logger to return the correct logger instance based on name
    def setup_logger_side_effect(name, *args, **kwargs):
        if name == BenchmarkRunner.__name__:
            return mock_runner_logger
        elif name.startswith("MockTaskHandler"): # Assuming handler class name
            return mock_handler_logger
        elif name == "Evaluator":
            return mock_evaluator_logger
        return MagicMock()
    mock_setup_logger_sut.side_effect = setup_logger_side_effect
    
    # Configure log_interval to be invalid
    config_with_invalid_log = basic_benchmark_config.model_copy(deep=True)
    config_with_invalid_log.evaluation.log_interval = -1 # Invalid value

    runner = BenchmarkRunner(config=config_with_invalid_log)
    runner.logger = mock_runner_logger # Ensure runner uses the main mock logger

    mock_handler = MagicMock(name="MockTaskHandler")
    mock_handler.logger = mock_handler_logger # Assign logger to handler
    mock_handler.process_batch.return_value = (["pred1"], ["label1"]) # Simulate valid batch processing
    
    mock_data_loader = [(["data1"], ["label1"])] # Simulate a DataLoader with one batch
    mock_evaluator = MagicMock(spec=Evaluator)
    mock_evaluator.logger = mock_evaluator_logger
    mock_evaluator._metric_instances_with_config = [] # No metrics to simplify

    success = runner._process_task_batches(mock_handler, mock_data_loader, "test_task_invalid_log", mock_evaluator)

    assert success is True
    mock_runner_logger.warning.assert_any_call(
        "Invalid log_interval value '-1' in evaluation config for task 'test_task_invalid_log'. Disabling intermediate metric logging."
    )
    # Ensure intermediate logging was not attempted (because interval is effectively 0)
    # This is implicitly tested by not having metrics and checking no error occurs.
    # A more direct test would involve metrics and checking their .result() wasn't called for intermediate logs.

@patch('src.benchmark.benchmark_loader.setup_logger')
def test_process_task_batches_handler_process_batch_exception(mock_setup_logger_sut, basic_benchmark_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    
    runner = BenchmarkRunner(config=basic_benchmark_config) # log_interval is 1 by default here
    runner.logger = mock_logger_instance

    mock_handler = MagicMock(name="MockTaskHandler")
    mock_handler.process_batch.side_effect = Exception("Handler processing error")
    
    mock_data_loader = [(["data1"], ["label1"])]
    mock_evaluator = MagicMock(spec=Evaluator)
    mock_evaluator._metric_instances_with_config = []

    success = runner._process_task_batches(mock_handler, mock_data_loader, "test_task_handler_fail", mock_evaluator)

    assert success is False # Should indicate failure
    mock_logger_instance.error.assert_called_once()
    args, _ = mock_logger_instance.error.call_args
    assert "Error processing batch 1 for task 'test_task_handler_fail': Handler processing error" in args[0]
    mock_evaluator.update_batch_metrics.assert_not_called()


@patch('src.benchmark.benchmark_loader.setup_logger')
def test_process_task_batches_prediction_label_mismatch(mock_setup_logger_sut, basic_benchmark_config):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_handler = MagicMock(name="MockTaskHandler")
    # Simulate handler returning mismatched predictions and labels
    mock_handler.process_batch.return_value = (["pred1", "pred2"], ["label1"]) 
    
    mock_data_loader = [(["data1"], ["label1"])] # Single batch for simplicity
    mock_evaluator = MagicMock(spec=Evaluator)
    mock_evaluator._metric_instances_with_config = []

    success = runner._process_task_batches(mock_handler, mock_data_loader, "test_task_mismatch", mock_evaluator)

    assert success is True # The method itself doesn't fail, but logs a warning
    mock_logger_instance.warning.assert_any_call(
        "Batch 1 for task 'test_task_mismatch': predictions (2) and labels (1) length mismatch. Skipping batch for evaluation."
    )
    mock_evaluator.update_batch_metrics.assert_not_called() # Metric update should be skipped

# --- Additional tests for _run_task_evaluation ---

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.TaskHandlerFactory')
def test_run_task_evaluation_handler_factory_fails(
    mock_task_handler_factory, mock_setup_logger_sut, basic_benchmark_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_task_handler_factory.get_handler.side_effect = ValueError("Handler factory error")
    
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_dataset_info = {"dataset": MagicMock(spec=Dataset), "name": "dummy", "task_config": dummy_task_config_obj, "streaming": False}

    results = runner._run_task_evaluation(mock_model, mock_tokenizer, dummy_task_config_obj, mock_dataset_info)

    assert results == {"error": "Handler not found: Handler factory error"}
    mock_logger_instance.error.assert_called_with(
        f"Could not get handler for task '{dummy_task_config_obj.name}' (type: {dummy_task_config_obj.type}): Handler factory error"
    )

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.DataLoader', side_effect=TypeError("DataLoader creation failed")) # Mock DataLoader to fail
@patch('src.benchmark.benchmark_loader.TaskHandlerFactory')
def test_run_task_evaluation_dataloader_creation_fails(
    mock_task_handler_factory, mock_dataloader, mock_setup_logger_sut, 
    basic_benchmark_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_handler_instance = MagicMock()
    mock_task_handler_factory.get_handler.return_value = mock_handler_instance
    
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    # Simulate a dataset that might cause DataLoader to fail (though failure is mocked directly)
    mock_dataset_info = {"dataset": "not_a_real_dataset_obj", "name": "dummy", "task_config": dummy_task_config_obj, "streaming": False} 

    results = runner._run_task_evaluation(mock_model, mock_tokenizer, dummy_task_config_obj, mock_dataset_info)

    assert results == {"error": "DataLoader creation failed: DataLoader creation failed"}
    mock_logger_instance.error.assert_called_with(
        f"Failed to create DataLoader for task '{dummy_task_config_obj.name}': DataLoader creation failed.", exc_info=True
    )

@patch('src.benchmark.benchmark_loader.setup_logger')
@patch('src.benchmark.benchmark_loader.BenchmarkRunner._process_task_batches', return_value=False) # Simulate batch processing failure
@patch('src.benchmark.benchmark_loader.TaskHandlerFactory')
@patch('src.benchmark.benchmark_loader.DataLoader') # Mock DataLoader to avoid its actual creation logic
@patch('src.benchmark.benchmark_loader.Evaluator') # Mock Evaluator
def test_run_task_evaluation_process_task_batches_fails(
    mock_evaluator_class, mock_dataloader_class, mock_task_handler_factory, 
    mock_process_batches, mock_setup_logger_sut,
    basic_benchmark_config, dummy_task_config_obj
):
    mock_logger_instance = MagicMock()
    mock_setup_logger_sut.return_value = mock_logger_instance
    runner = BenchmarkRunner(config=basic_benchmark_config)
    runner.logger = mock_logger_instance

    mock_handler_instance = MagicMock(name="MockHandler")
    mock_task_handler_factory.get_handler.return_value = mock_handler_instance
    
    # Mock DataLoader instance
    mock_dataloader_instance = MagicMock(name="MockDataLoader")
    mock_dataloader_class.return_value = mock_dataloader_instance

    # Mock Evaluator instance
    mock_evaluator_instance = MagicMock(name="MockEvaluator")
    mock_evaluator_instance.finalize_results.return_value = {"metric_partial": 0.5} # Simulate some partial results
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model = MagicMock(name_or_path="test_model_id") # Give it a name_or_path
    mock_tokenizer = MagicMock()
    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_info = {"dataset": mock_dataset_obj, "name": "dummy", "task_config": dummy_task_config_obj, "streaming": False} 

    results = runner._run_task_evaluation(mock_model, mock_tokenizer, dummy_task_config_obj, mock_dataset_info)
    
    # Even if _process_task_batches returns False, _run_task_evaluation should still call finalize_results
    mock_evaluator_instance.finalize_results.assert_called_once()
    assert results == {"metric_partial": 0.5} # It should return whatever finalize_results gave
    mock_logger_instance.warning.assert_any_call(
         f"Task '{dummy_task_config_obj.name}' on model 'test_model_id' processing was not fully successful due to batch errors."
    )