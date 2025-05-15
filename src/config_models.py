from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class GeneralConfig(BaseModel):
    # General parameters for the benchmark
    # e.g., benchmark name, output directory, random seed
    experiment_name: str = "Benchmark Experiment"
    output_dir: Path = Path("./benchmarks")
    random_seed: Optional[int] = 42

class ReportingConfig(BaseModel):
    # Reporting parameters for the benchmark
    # e.g., report format, enabled metrics, etc.
    enabled: bool = True
    format: str = "json"
    leaderboard_enabled: bool = False
    generate_visuals: Optional[Dict[str, bool]] = None
    output_dir: Path = Path("./reports")

class AdvancedConfig(BaseModel):
    # Advanced parameters for the benchmark
    # e.g., multi-GPU, TPU, distributed training, etc.
    enable_multi_gpu: bool = False
    use_tpu: bool = False
    distributed_training: bool = False
    batch_size: int = 32
    # Add other advanced params used in TaskHandler (e.g., generate_max_length etc.)
    truncation: bool = True
    padding: bool = True
    generate_max_length: int = 512
    skip_special_tokens: bool = True
    max_new_tokens: int = 50
    num_beams: int = 1
    do_sample: bool = False
    use_cache: bool = False
    clean_up_tokenization_spaces: bool = True

class DatasetConfig(BaseModel):
    # Configuration for datasets
    # e.g., dataset name, type, split, etc.
    name: str
    source_type: str = "hf_hub" # Added default
    config: Optional[str] = None
    split: str = "validation"
    data_dir: Optional[Path] = None
    streaming: bool = True
    dataset_specific_fields: Optional[Dict[str, str]] = None
    loader_args: Dict[str, Any] = Field(default_factory=dict)
    max_samples: Optional[int] = None

class MetricConfig(BaseModel):
    # Configuration for evaluation metrics
    # e.g., metric name, type, parameters, etc.
    name: str
    options: Dict[str, Any] = Field(default_factory=dict)

class TaskConfig(BaseModel):
    # Configuration for tasks
    # e.g., task name, type, datasets, evaluation metrics, etc.
    name: str
    type: str # e.g., "classification", "generation"
    description: Optional[str] = None
    datasets: List[DatasetConfig]
    evaluation_metrics: List[MetricConfig]
    handler_options: Optional[Dict[str, Any]] = None

class ModelConfig(BaseModel):
    # Configuration for models
    # e.g., model name, type, framework, checkpoint, etc.
    name: str
    framework: str
    checkpoint: Optional[str] = None
    variant: Optional[str] = None
    size: Optional[str] = None
    quantization: Optional[str] = None
    offloading: Optional[bool] = None

class ModelParamsConfig(BaseModel):
    # Configuration for model parameters
    # e.g., model-specific parameters, tokenizer settings, etc.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # Add others...

class EvaluationConfig(BaseModel):
    # Configuration for evaluation
    # e.g., evaluation parameters, batch size, etc.
    log_interval: Optional[int]

class BenchmarkConfig(BaseModel):
    # Main configuration class for the benchmark
    # e.g., general parameters, tasks, models, evaluation, reporting, etc.
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    evaluation: Optional[EvaluationConfig] = Field(default_factory=EvaluationConfig)
    tasks: List[TaskConfig]
    models: List[ModelConfig]
    model_parameters: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)