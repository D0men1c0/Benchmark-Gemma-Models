import pytest
from pathlib import Path
import yaml
from pydantic import ValidationError

from src.config_models import (
    GeneralConfig,
    DatasetConfig,
    MetricConfig,
    TaskConfig,
    ModelConfig,
    BenchmarkConfig,
    AdvancedConfig,
    ReportingConfig,
    EvaluationConfig,
    ModelParamsConfig
)

# --- Simpler Tests for Config Models ---
def test_general_config_defaults():
    """Test that GeneralConfig can be initialized with defaults."""
    config = GeneralConfig()
    assert config.experiment_name == "Benchmark Experiment"
    assert config.output_dir == Path("./benchmarks")
    assert config.random_seed == 42

def test_dataset_config_minimal():
    """Test DatasetConfig with minimal required fields."""
    config = DatasetConfig(name="test_dataset")
    assert config.name == "test_dataset"
    assert config.source_type == "hf_hub"
    assert config.split == "validation"
    assert config.streaming is True

def test_metric_config_minimal():
    """Test MetricConfig with minimal required fields."""
    config = MetricConfig(name="accuracy")
    assert config.name == "accuracy"
    assert config.options == {}

def test_model_config_torch_dtype_validation():
    """Test ModelConfig torch_dtype validation."""
    ModelConfig(name="m1", framework="hf", torch_dtype="float16")
    ModelConfig(name="m2", framework="hf", torch_dtype="bfloat16")
    ModelConfig(name="m3", framework="hf", torch_dtype="float32")
    ModelConfig(name="m4", framework="hf", torch_dtype=None)
    with pytest.raises(ValidationError):
        ModelConfig(name="m_invalid", framework="hf", torch_dtype="int8")

def test_dataset_config_custom_script_fields():
    """Test DatasetConfig fields relevant to custom scripts."""
    data = {
        "name": "my_custom_ds",
        "source_type": "custom_script",
        "script_path": "path/to/script.py",
        "function_name": "load_my_data",
        "split": "train",
        "script_args": {"arg1": "value1"}
    }
    config = DatasetConfig(**data)
    assert config.script_path == "path/to/script.py"
    assert config.function_name == "load_my_data"
    assert config.script_args == {"arg1": "value1"}

# --- Loading and Validating YAML Configurations ---

@pytest.fixture
def project_root_dir():
    """Fixture to get the project root directory dynamically."""
    return Path(__file__).parent.parent # tests/ -> tests/benchmark -> tests -> root

@pytest.fixture
def basic_config_yaml_content(project_root_dir):
    """Loads content from basic_benchmark_config.yaml."""
    #
    config_path = project_root_dir / "src" / "config" / "basic_benchmark_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def advanced_custom_config_yaml_content(project_root_dir):
    """Loads content from advanced_custom_benchmark_config.yaml."""
    config_path = project_root_dir / "src" / "config" / "advanced_custom_benchmark_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def default_config_yaml_content(project_root_dir):
    """Loads content from the main benchmark_config.yaml."""
    config_path = project_root_dir / "src" / "config" / "benchmark_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_load_basic_benchmark_config_from_yaml(basic_config_yaml_content):
    """Test loading and validation of the basic_benchmark_config.yaml content."""
    config = BenchmarkConfig.model_validate(basic_config_yaml_content)
    assert config.general.experiment_name == "Gemma_Benchmark_Comprehensive_2025"
    assert len(config.tasks) == 1
    assert config.tasks[0].name == "MMLU (all Subset - Templated)"
    assert config.tasks[0].type == "multiple_choice_qa"
    assert config.tasks[0].datasets[0].name == "cais/mmlu"
    assert config.tasks[0].datasets[0].config == "all"
    assert config.tasks[0].datasets[0].max_samples == 50
    assert len(config.models) == 1
    assert config.models[0].name == "gemma-2b"
    assert config.models[0].framework == "huggingface"
    assert config.models[0].checkpoint == "google/gemma-2b"
    assert config.models[0].quantization == "4bit"
    assert config.advanced.batch_size == 10
    assert config.reporting.output_dir == Path("./reports")

def test_load_advanced_custom_benchmark_config_from_yaml(advanced_custom_config_yaml_content):
    """Test loading and validation of the advanced_custom_benchmark_config.yaml content."""
    config = BenchmarkConfig.model_validate(advanced_custom_config_yaml_content)
    assert config.general.experiment_name == "Comprehensive_Benchmark_Suite_EN"
    assert len(config.tasks) == 3

    # Check a custom model
    custom_model_found = False
    for model_cfg in config.models:
        if model_cfg.name == "gpt2-custom-script-loader":
            custom_model_found = True
            assert model_cfg.framework == "custom_script"
            assert model_cfg.script_path == "src/benchmark/custom_loader_scripts/my_model_functions.py"
            assert model_cfg.function_name == "load_local_hf_model_example"
            assert model_cfg.checkpoint == "gpt2"
            break
    assert custom_model_found, "Custom script model 'gpt2-custom-script-loader' not found in advanced config"

    # Check a custom script task
    custom_task_found = False
    for task_cfg in config.tasks:
        if task_cfg.name == "CreativeThemeElaboration_EN":
            custom_task_found = True
            assert task_cfg.type == "custom_script"
            assert task_cfg.datasets[0].source_type == "custom_script"
            assert task_cfg.datasets[0].script_path == "src/benchmark/custom_loader_scripts/my_dataset_functions.py"
            assert task_cfg.datasets[0].script_args["data_file_path"] == "src/benchmark/custom_loader_scripts/data/sample_train_dataset.csv"
            assert task_cfg.handler_options["handler_script_path"] == "src/benchmark/custom_loader_scripts/my_task_handler_functions.py"
            assert task_cfg.handler_options["postprocessor_key"] == "custom_script"
            # Check a custom metric within this task
            custom_metric_found = any(
                metric.name == "custom_script" and 
                metric.options.get("metric_script_path") == "src/benchmark/custom_loader_scripts/my_metrics_functions.py" and
                metric.options.get("metric_result_function_name") == "result_avg_pred_length"
                for metric in task_cfg.evaluation_metrics
            )
            assert custom_metric_found, "Custom metric for average_elaboration_length not found in CreativeThemeElaboration_EN"
            break
    assert custom_task_found, "Custom task 'CreativeThemeElaboration_EN' not found in advanced config"
    
    assert config.evaluation.log_interval == 2
    assert config.advanced.batch_size == 4

def test_load_default_benchmark_config_from_yaml(default_config_yaml_content):
    """Test loading and validation of the main benchmark_config.yaml content."""
    config = BenchmarkConfig.model_validate(default_config_yaml_content)
    assert config.general.experiment_name == "Gemma_Benchmark_Comprehensive_2025"
    assert len(config.tasks) > 0 # Assuming there are tasks defined in the default config
    assert len(config.models) > 0

    # Example check for a specific task and model from benchmark_config.yaml
    mmlu_task_present = any(task.name == "MMLU (all Subset - Templated)" for task in config.tasks)
    assert mmlu_task_present, "MMLU task not found in default benchmark_config.yaml"
    
    gemma_7b_present = any(model.name == "gemma-7b" and model.checkpoint == "google/gemma-7b" for model in config.models)
    assert gemma_7b_present, "gemma-7b model not found in default benchmark_config.yaml"
    
    # Check some advanced parameters
    assert config.advanced.batch_size == 10
    assert config.advanced.max_new_tokens == 150
    assert config.evaluation.log_interval == 5