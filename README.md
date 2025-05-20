# Benchmark-Gemma-Models [![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-green)](https://www.python.org/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D0men1c0/Benchmark-Gemma-Models/blob/main/hello_world.ipynb)

[![Powered by Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)](https://huggingface.co/transformers) [![Uses PyTorch](https://img.shields.io/badge/Uses-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/) [![Uses TensorFlow](https://img.shields.io/badge/Uses-TensorFlow-FF6F00?logo=tensorflow)](https://www.tensorflow.org/) [![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io)

## Unleash Your LLM Benchmarking: Extreme Flexibility & Resource Mastery

This framework is engineered for comprehensive, deeply customizable, and resource-conscious evaluation of Large Language Models (LLMs), with a primary focus on **Gemma models** and compatibility with **LLaMA**, **Mistral**, and other architectures. What sets this suite apart is its **unparalleled flexibility** through custom script integration for every core component (models, datasets, tasks, metrics, post-processing) and its meticulous **attention to resource optimization**, enabling extensive benchmarks even on constrained hardware.

Whether you're a researcher experimenting with novel evaluation techniques, a developer fine-tuning models, or an enthusiast comparing the latest LLMs, this suite provides a robust, automated, and scalable solution. Go beyond standard benchmarks by defining your own logic, seamlessly integrate models from diverse sources including PyTorch, TensorFlow, and custom environments, and gain granular insights with advanced logging and intelligent output processing, all while maintaining efficient memory and disk utilization.

## Table of Contents
- [Overview](#overview)
- [Modular Architecture](#modular-architecture)
  - [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Examples & Getting Started](#examples--getting-started)
  - [Quick Test with Colab](#quick-test-with-colab)
  - [Generating a Default Configuration](#generating-a-default-configuration)
  - [Visualizing Benchmark Results](#visualizing-benchmark-results)
- [Configuring Benchmarks (`benchmark_config.yaml`)](#configuring-benchmarks-benchmark_configyaml)
  - [Leveraging Custom Scripts for Ultimate Flexibility](#leveraging-custom-scripts-for-ultimate-flexibility)
- [Workflow Deep Dive](#workflow-deep-dive)
- [Core Components](#core-components)
- [Key Advantages & Why Use This Benchmark Suite?](#key-advantages--why-use-this-benchmark-suite)
- [Next Steps](#next-steps)
- [License and Important Disclaimers](#license-and-important-disclaimers)

## Overview

This framework offers a robust, automated, and scalable solution to assess the performance of Large Language Models (LLMs), with a primary focus on **Gemma models**. It provides insights across diverse tasks, including standard academic benchmarks and custom datasets. The system emphasizes scalability through efficient batch processing, stateful metrics, and intelligent resource management; extensibility via a modular factory-based architecture that now fully supports custom user scripts for all core components; and ease of use, ensuring researchers can efficiently measure, compare, and reproduce model performance with unprecedented control.

---

## Modular Architecture

The architecture is designed for clarity, maintainability, and extreme extensibility. Key components are managed through factories, allowing for easy integration of new models (from Hugging Face, PyTorch, TensorFlow, or custom loading scripts), datasets (from Hugging Face Hub, local files, or custom data loading functions), tasks (including custom task handlers), prompt strategies, post-processing routines (including regex-based and custom script-based), and evaluation metrics (including custom script-based metrics). Configuration is centralized using Pydantic models loaded from YAML files.

### Directory Structure
```bash
├── 📦 src/                     # Main source code
│   ├── 📂 benchmark/           # Core benchmarking logic (models, datasets, tasks, evaluation, etc.)
│   │   ├── 📂 dataset/         # Dataset loading, normalization, and custom data functions
│   │   ├── 📂 evaluation/      # Metrics computation and evaluation orchestration
│   │   ├── 📂 models/          # Model loading (HF, PyTorch, TF, Custom) and quantization
│   │   ├── 📂 postprocessing/  # Output cleaning and structuring
│   │   ├── 📂 prompting/       # Prompt engineering and building strategies
│   │   ├── 📂 reporting/       # Saving benchmark results
│   │   └── 📂 tasks/           # Task-specific logic and handlers
│   ├── 📂 config/              # YAML configuration files for benchmarks
│   ├── 📂 custom_loader_scripts/ # User-defined Python scripts for custom components
│   │   └── 📂 data/            # Sample data for custom loader examples
│   ├── 📂 scripts/             # Execution entry points (e.g., run_benchmark.py)
│   ├── 📂 utils/               # Shared utilities (e.g., logger)
│   ├── config_models.py        # Pydantic models for configuration validation
│   └── generate_default_config.py # Script to create a default config file
│
├── 📦 visualize/               # Scripts and dashboard for results visualization
│   ├── dashboard.py            # Streamlit dashboard application
│   ├── data_utils.py           # Data processing for the dashboard
│   └── plotting_results.ipynb  # Jupyter notebook for plotting
│
├── 📓 hello_world.ipynb        # Quick start/demo notebook for Colab
├── 📄 LICENSE                  # Apache 2.0 License
└── 📄 README.md                # This file
```

---

## Quick Start

### 1. Clone Repository
```
git clone https://github.com/D0men1c0/Benchmark-Gemma-Models
cd Benchmark-Gemma-Models
```
### 2. Installation Dependencies
```bash
pip install -r requirements.txt
```
### 3. Setup Environment
```bash
cd src
python -m venv venv
source venv/bin/activate  # Linux/Mac
# For Windows PowerShell:
# venv\Scripts\Activate.ps1
# For Windows CMD:
# venv\Scripts\activate.bat
cd ..
```
### 4. Run Benchmark

To run with the default comprehensive configuration:

```bash
python src/scripts/run_benchmark.py \
    --config src/config/benchmark_config.yaml \
    --output-dir results/ \
    --log-level INFO 
```

To run with the advanced custom configuration showcasing custom script loaders:

```bash
python src/scripts/run_benchmark.py \
    --config src/config/advanced_custom_benchmark_config.yaml \
    --output-dir results_custom/ \
    --log-level DEBUG
```
The framework features detailed logging, writing to a timestamped file in `run_logs/` by default (e.g., `run_logs/benchmark_YYYYMMDD_HHMMSS.log`)

---

## Examples & Getting Started

### Quick Test with Colab
A `hello_world.ipynb` notebook is provided at the root of the repository. It offers a simple way to clone the repository, install dependencies, and run a basic benchmark directly in a Google Colab environment. This is a great starting point for quickly testing the framework. Note its capability to run even larger models like 13B parameters on a free Colab T4 GPU when 4-bit quantization is enabled.

### Generating a Default Configuration
To get started with a base configuration, you can use the `generate_default_config.py` script located in the `src/` directory:

```bash
python src/generate_default_config.py
```

This will create a `default_benchmark_config.yaml` file in `src/config/` which you can then customize.

### Visualizing Benchmark Results
An interactive dashboard is available to explore the benchmark results.
1.  Ensure you have run a benchmark and results are saved (e.g., in `benchmarks_output/benchmark_results.json`).
2.  Run the Streamlit application:

```bash
streamlit run visualize/dashboard.py
```

This will launch the dashboard in your web browser. The `visualize/plotting_results.ipynb` notebook also contains examples of how benchmark data can be plotted programmatically.

Here's a sneak peek of the dashboard in action:

**Overall Dashboard Interface:**

![Overall Dashboard Interface](img/Plots/dashboard.png)

**Example: MMLU Exact Match Comparison Across Models:**

![MMLU Exact Match](img/Plots/MMLU-exact-match.png)

---

## Configuring Benchmarks (`benchmark_config.yaml`)

All benchmark experiments are defined in a YAML configuration file (e.g., `src/config/benchmark_config.yaml` or `src/config/advanced_custom_benchmark_config.yaml`). This file details the models, tasks, datasets, metrics, and runtime settings. You can generate a full default configuration using `python src/generate_default_config.py`.

Here's a minimal example focusing on standard Hugging Face usage (refer to `src/config/basic_benchmark_config.yaml` for a runnable version):

```yaml
general:
  experiment_name: "Gemma_Benchmark_Comprehensive_2025"
  output_dir: "./benchmarks_output"
  random_seed: 42

tasks:
  - name: "MMLU (all Subset - Templated)"
    type: "multiple_choice_qa"
    datasets:
      - name: "cais/mmlu"
        source_type: "hf_hub"
        config: "anatomy" # Specific MMLU subject
        split: "validation"
        max_samples: 10 # For a very quick test
    handler_options:
      prompt_builder_type: "mmlu"
      prompt_template: # Prompt template can be defined here
    evaluation_metrics:
      - name: exact_match
        options: { normalize: true, ignore_case: true, ignore_punct: false }

models:
  - name: "gemma-2b - 4quant"
    framework: "huggingface"
    checkpoint: "google/gemma-2b" # Or other models like phi-2, llama3
    quantization: "4bit"
    torch_dtype: "bfloat16" # Recommended for Gemma with quantization
    offloading: true # Useful for memory management

advanced:
  batch_size: 5     # Smaller batch size for limited VRAM
  max_new_tokens: 5 # MMLU expects a single letter
```

**Key sections briefly:**
* **`general`**: Experiment-wide settings.
* **`tasks`**: List of tasks. Each specifies its `type`, `datasets` (with `max_samples`), `handler_options` (like `prompt_builder_type` and optional `prompt_template`), and `evaluation_metrics`.
* **`models`**: List of models, detailing `checkpoint`, `framework` (e.g., `huggingface`, `pytorch`, `tensorflow`, `custom_script`), `quantization`, etc.
* **`evaluation`**: Settings for the evaluation process (e.g., `log_interval`).
* **`reporting`**: How to save results (e.g., `format`).
* **`advanced`**: Global parameters for tokenization and generation (e.g., `batch_size`, `max_new_tokens`).

---

## Leveraging Custom Scripts for Ultimate Flexibility

A powerful feature of this framework is its ability to integrate custom Python scripts for nearly every component of the benchmarking pipeline. This allows for unparalleled flexibility beyond standard configurations.

The `src/config/advanced_custom_benchmark_config.yaml` file provides a comprehensive example of how to use these custom loaders. Here’s a snippet illustrating how to define a custom model loader and a custom task:

```yaml
models:
  - name: "gpt2-custom-script-loader"
    framework: "custom_script" # Signals to use CustomScriptModelLoader
    script_path: "src/benchmark/custom_loader_scripts/my_model_functions.py" # Path to your script
    function_name: "load_local_hf_model_example" # Function to call within the script
    # Arguments below are passed to your custom function:
    checkpoint: "gpt2" # Or path to local model files
    torch_dtype_str: "float32" # Custom arg for your script
    # ... other args your script might need

tasks:
  - name: "CreativeThemeElaboration_EN"
    type: "custom_script" # Uses CustomScriptTaskHandler
    description: "Evaluate creative elaboration using custom logic."
    datasets:
      - name: "sample_themes_for_elaboration_en"
        source_type: "custom_script" # Uses CustomScriptDatasetLoader
        script_path: "src/benchmark/custom_loader_scripts/my_dataset_functions.py" #
        function_name: "load_simple_csv_data"
        split: "train"
        max_samples: 8
        script_args: # Arguments specific to your dataset loading function
            data_file_path: "src/benchmark/custom_loader_scripts/data/sample_train_dataset.csv"

    handler_options: # Options passed to the CustomScriptTaskHandler
        handler_script_path: "src/benchmark/custom_loader_scripts/my_task_handler_functions.py"
        handler_function_name: "creative_elaboration_processor" # Your function to process a batch
        prompt_template: "Expand on this idea: {theme}" # Can still use templates
        postprocessor_key: "custom_script" # Use CustomScriptPostProcessor
        postprocessor_script_path: "src/benchmark/custom_loader_scripts/my_post_processor_functions.py" #
        postprocessor_function_name: "clean_elaboration_output" #
    evaluation_metrics:
       - name: "rouge" # Standard metric
       - name: "custom_script" # Use CustomScriptMetric
         options:
           metric_script_path: "src/benchmark/custom_loader_scripts/my_metrics_functions.py" #
           metric_init_function_name: "init_avg_pred_length_state"
           metric_update_function_name: "update_avg_pred_length_state"
           metric_result_function_name: "result_avg_pred_length"
           metric_script_args:
            result_key_name: "average_elaboration_length"
```

This demonstrates the ability to define custom loading and processing logic for datasets, models, task execution, post-processing, and metrics, all configured via YAML and executed by the corresponding `CustomScript*` components.

For the complete structure and all available options, please refer to `src/config/benchmark_config.yaml`, `src/config/advanced_custom_benchmark_config.yaml`, or the file generated by `src/generate_default_config.py`.

## Workflow Deep Dive

![Benchmark Pipeline Diagram](img/mermaid_new_diagram2.png)

The benchmarking process is managed by `BenchmarkRunner`, initialized via `run_benchmark.py`. It follows a modular and extensible pipeline:

1.  **Config Loading**
    * Loads `benchmark_config.yaml` using Pydantic for type safety.
    * Governs all aspects: models, datasets, tasks, metrics, runtime settings, output options.

2.  **Dataset Preparation**
    * Uses `DatasetFactory` to load datasets from Hugging Face, local files, or custom scripts.
    * Applies automatic field normalization to standardize data formats across tasks.

3.  **Model Loading**
    * `ModelLoaderFactory` loads models via Hugging Face, PyTorch, TensorFlow, or custom scripts.
    * Supports quantization (4-bit, 8-bit) and memory-efficient loading/offloading.
    * Optional Hugging Face cache cleanup is available (disabled by default, use with caution).

4.  **Prompt Building and Task Execution**
    * Prompts are constructed using `PromptBuilderFactory` based on task config.
    * `TaskHandler` runs generation or classification, applying task-specific logic and generation settings.

5.  **Post-processing**
    * `PostProcessorFactory` cleans and extracts final answers (e.g., letter choice, numerical values).
    * Ensures outputs are in a comparable format for evaluation.

6.  **Evaluation**
    * `Evaluator` computes stateful metrics batch-by-batch using `MetricFactory`.
    * Supports both built-in and custom metrics.
    * Final results are aggregated after all batches are processed.

7.  **Reporting**
    * Results are saved in user-defined formats (JSON, CSV, PDF) via `FileManager`.

This design ensures the process is robust, reproducible, and highly customizable across tasks, models, and evaluation strategies.

---

## Core Components

| Component                      | Description                                                                                                                                                                                             | Path (`src/`)                                                                |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------- |
| `run_benchmark.py`             | Main script to execute benchmarks based on a provided configuration.                                                                                                                                      | `scripts/run_benchmark.py`                                                   |
| `benchmark_config.yaml`        | Central YAML file (Pydantic-validated via `config_models.py`) defining experiments: models, tasks, datasets, metrics, and runtime settings.                                                               | `config/benchmark_config.yaml`                                               |
| `BenchmarkRunner`              | Orchestrates the entire benchmark lifecycle: loading components, iterating through tasks and models, managing evaluation, and reporting results.                                                          | `benchmark/benchmark_loader.py`                                              |
| `Model Loading` | `ModelLoaderFactory` and concrete/custom model loaders manage the loading of LLMs from Hugging Face, PyTorch, TensorFlow, or custom scripts. Supports quantization and offloading.                      | `benchmark/models/`                                                          |
| `Dataset Loading (Data Loaders)` | `DatasetFactory` and concrete/custom dataset loaders handle fetching and preparing data from Hugging Face Hub, local files, or custom scripts. Includes field normalization and streaming.              | `benchmark/dataset/`                                                         |
| `Task Handling` | `TaskHandlerFactory` and concrete/custom task handlers manage the specific logic for each benchmark task, including model interaction and batch processing.                                             | `benchmark/tasks/`                                                           |
| `Prompt Building` | `PromptBuilderFactory` and concrete prompt builders construct the precise input prompts for models based on task configuration and templates.                                                           | `benchmark/prompting/`                                                       |
| `Post-processing` | `PostProcessorFactory` and concrete/custom post-processors clean and structure raw model outputs into a format suitable for evaluation (e.g., using regex or custom logic).                               | `benchmark/postprocessing/`                                                  |
| `Metric Evaluation` | `MetricFactory` and concrete/custom metrics implement various evaluation measures. The `Evaluator` manages their stateful, batch-by-batch computation.                                                 | `benchmark/evaluation/`                                                      |
| `custom_loader_scripts/`       | Key directory for user-defined Python scripts. Houses custom logic for all extensible components: models, datasets, task handlers, post-processors, and metrics.                                        | `benchmark/custom_loader_scripts/`                                           |
| `FileManager`                  | Responsible for saving the final, aggregated benchmark results into user-specified formats (e.g., JSON, CSV, PDF).                                                                                        | `benchmark/reporting/file_manager.py`                                      |
| `visualize/dashboard.py`       | An interactive Streamlit application for visualizing and exploring the benchmark results saved by the `FileManager`.                                                                                        | `visualize/dashboard.py` (Note: This is outside `src/` but key for results)  |

---

## Key Advantages & Why Use This Benchmark Suite?

This benchmark suite is built for robust, flexible, and resource-efficient evaluation of LLMs, with a focus on extensibility, modularity, and developer-friendliness.

### Ultimate Extensibility with Custom Scripts
Integrate your own Python functions for every component:
- **Models**: Load from any source (local, proprietary, custom formats).
- **Datasets**: Ingest non-standard or private data formats.
- **Tasks**: Customize logic for prompt generation or interaction.
- **Post-Processing**: Clean or parse outputs however needed.
- **Metrics**: Create evaluation metrics tailored to your use case.

All configurable via YAML using `framework: custom_script`, `source_type: custom_script`, etc.

### Centralized, Pydantic-Validated Configuration
Use `benchmark_config.yaml` or `advanced_custom_benchmark_config.yaml` to define:
- **Model setup** (LLM name, quantization, dtype, offloading)
- **Task definitions** (type, prompt templates, post-processing)
- **Dataset loading** (Hugging Face, local, or custom)
- **Evaluation and runtime settings** (metrics, generation parameters)

The configuration is strongly typed and validated through Pydantic models.

### Broad Task and Metric Support (Built-in and Custom)
Supported tasks include:
- `multiple_choice_qa`, `summarization`, `translation`, `classification`, `math_reasoning_generation`, and `custom_script`

Supported metrics:
- **Accuracy, F1, ROUGE, BERTScore, BLEU, METEOR**
- **Semantic similarity, Perplexity, Toxicity**
- Custom metrics via Python scripts

### Efficient and Scalable Design
- Models and datasets are loaded on demand to minimize memory usage.
- Supports quantization (4-bit, 8-bit) to run large models on limited hardware.
- Stateful metric evaluation reduces memory footprint during large-scale runs.
- Optional Hugging Face disk cache cleanup (experimental, disabled by default).

### Modular "Factory-First" Architecture
Uses factory classes for models, datasets, prompts, tasks, post-processors, and metrics.  
Adding a new component only requires subclassing and registration.

### Results Visualization and Logging
- Timestamped logs saved in `run_logs/`
- Output formats: JSON, CSV, PDF
- Streamlit dashboard and Jupyter notebooks for result exploration

### Practical and Reproducible
- Designed to work under constrained conditions (e.g., Colab T4)
- YAML-driven configuration ensures experiments are easy to replicate and share
- Includes `generate_default_config.py` to quickly create a base config

### Framework Support
- Native integration with Hugging Face Transformers, PyTorch, TensorFlow
- Full support for custom loading logic via Python scripts

---

## Next Steps

We aim to strengthen the framework’s stability, usability, and reach, while welcoming community contributions and feedback.

### Short- to Mid-Term Goals

- **Improve Testing and Robustness**
  - Add unit and integration tests for key components (models, datasets, tasks, metrics, post-processing).
  - Validate the full pipeline across varied configurations, especially with custom scripts.

- **Expand Metrics and Benchmarks**
  - Add more standard evaluation metrics.
  - Make it easier to plug in new academic benchmarks.

- **Enhance Documentation and Examples**
  - Create richer examples for custom scripts.
  - Develop complete GitHub Pages documentation.

- **Refine GLUE Task Support**
  - Improve prompt templates and output parsing for GLUE-related tasks.

- **Engage the Community**
  - Collect feedback, fix pain points, and encourage external contributions.

### Long-Term Vision

- **Comprehensive Documentation Site**
  - Publish a full-featured documentation hub with tutorials and configuration guides.

- **Advanced Visualization**
  - Add features to the Streamlit dashboard: error analysis, historical comparisons, and model insights.

- **Cloud & Distributed Execution**
  - Provide examples or tools for running benchmarks on Colab Pro, AWS, GCP, etc.
  - Explore distributed benchmarking for large-scale evaluations.

- **Database Integration**
  - Support storing and querying benchmark results using SQLite/PostgreSQL for longitudinal analysis.

- **Wider Framework & Model Support**
  - Continuously add compatibility for new LLMs and serving backends.

---

## License and Important Disclaimers

**Project License:**

This project, Benchmark-Gemma-Models, is released under the **Apache 2.0 License**. You can find the full license text in the `LICENSE` file in the root of this repository.

**Important Disclaimers:**

* **Nature of Benchmark Results & User Responsibility:**
    All benchmark results provided by or visualized through this framework (e.g., via the dashboard or plots) are **experimental** and intended for **illustrative and demonstrative purposes only.** They are generated under specific, resource-constrained conditions (often using quantized models, small data subsets, limited prompting, and basic post-processing on hardware like Google Colab's free T4 GPU tier) and **have not been rigorously validated against official leaderboards.**

    Key limitations of our illustrative results include:
    * Use of **small data samples** (typically ~500 examples per task).
    * Employment of **quantized models** (4-bit/8-bit) to fit limited hardware.
    * Application of **basic prompting strategies** and **simplified post-processing** routines.
    * Imposition of **capped generation lengths** (e.g., 512 input / 150 output tokens).
    * Reliance on **standard metric implementations** (from `evaluate`, `nltk`, `bert-score`, etc.) which may differ from official evaluation setups.

    The maintainers of this project are not responsible for any decisions made or actions taken based on these experimental results. Users are solely responsible for interpreting and using any outputs. We encourage you to use these illustrative results as a starting point to explore the framework’s capabilities. **This suite is designed to empower you to run your own comprehensive experiments** under your specific conditions and with your chosen configurations to obtain benchmarks relevant to your needs.

* **Disk Cache Cleanup Disclaimer:**
    The experimental feature for cleaning up the Hugging Face model disk cache (`cleanup_model_cache_after_run`) involves deleting local directories. While safety checks are implemented within the framework, the actual file/directory deletion commands within the `_helper_disk_cache_cleanup` method in `src/benchmark/benchmark_loader.py` are **intentionally commented out by default.** Users wishing to use this feature must:
    1.  Carefully review the relevant code in `src/benchmark/benchmark_loader.py`.
    2.  Explicitly uncomment the deletion logic.
    3.  Understand and accept that they are enabling this feature **entirely at their own risk.**
    The project maintainers **are not and will not be responsible for any accidental data loss or damage** resulting from the activation or use of this disk cache cleanup feature. It is provided as-is, with the strong recommendation that users understand the implications before enabling it.

* **Models Used and Third-Party License Compliance:**
    This framework provides tools and scripts to evaluate third-party language models. It **does not distribute these models directly.** Users are responsible for obtaining access to these models and ensuring full compliance with their respective terms of use, licenses, and any applicable usage restrictions. Key model providers include, but are not limited to:
    * **Google Gemma:** Subject to [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
    * **Mistral Models:** Subject to the terms provided by Mistral AI (e.g., [Mistral AI Technology](https://mistral.ai/terms#terms-of-service)).
    * **Meta LLaMA Models:** Subject to the [Meta LLaMA License](https://ai.meta.com/llama/license/).
    * **Any other models used in this framework** are subject to their respective licenses and terms, which users must review and comply with prior to use.

* **No Affiliation or Endorsement:**
    This project is an independent initiative and is not affiliated with, sponsored by, or endorsed by Google, Mistral AI, Meta Platforms, Inc., or any other third-party model provider. All product names, logos, and brands are property of their respective owners.

---