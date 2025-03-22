# Gemma Model Benchmark Suite

[![Python: 3.12](https://img.shields.io/badge/Python-3.12%2B-green)](https://www.python.org/)

A modular framework for benchmarking Gemma models against academic benchmarks (MMLU, GSM8K) and custom datasets. Designed with separation of concerns and extensibility at its core.

---

## 🧱 Modular Architecture

### Directory Structure
```bash
📦 src
├── 📂 config/                 # Configuration management
│   ├── benchmark_config.yaml  # Main benchmark parameters
│   ├── logging.conf           # Logging configurations
│   └── prove123.yaml          # Example task-specific config
│
├── 📂 datasets/               # Dataset handling
│   ├── base_dataset_loader.py # Abstract dataset interface
│   ├── concrete_dataset_loader.py # Hub/S3/local implementations
│   └── dataset_factory.py     # Dataset loader factory
│
├── 📂 evaluation/             # Metrics & analysis
│   ├── base_metrics.py        # Metric interface
│   ├── concrete_metrics.py    # 15+ implemented metrics
│   └── metric_factory.py      # Metric computation orchestration
│
├── 📂 models/                 # Model management
│   ├── base_model_loader.py   # Abstract model interface
│   ├── concrete_models.py     # Hugging Face/TensorFlow/PyTorch loading
│   └── models_factory.py      # Model loading factory
│
├── 📂 scripts/                # Execution workflows
│   ├── benchmark_loader.py    # Core benchmarking logic
│   ├── evaluator.py           # Results analysis
│   └── run_benchmark.py       # Main entrypoint
│
├── 📂 tasks/                  # Task-specific logic
│   ├── base_task_handler.py   # Task interface
│   ├── concrete_task_handlers.py # Classification/Generation
│   └── task_handlers_factory.py # Task orchestration
│
├── 📂 utils/                  # Shared utilities
│   ├── file_manager.py        # Multi-format I/O (JSON/YAML/CSV/PDF)
│   └── logger.py              # Unified logging system
│
├── generate_default_config.py # Config generator
```

## Installation
```bash
pip install -r requirements.txt
