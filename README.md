# Benchmark-Gemma-Models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D0men1c0/Benchmark-Gemma-Models/blob/main/hello_world.ipynb) [![Read My Blog Post](https://img.shields.io/badge/Medium-Read_My_Blog_Post-black?logo=medium)](https://medium.com/@domenicolacavalla8/customizable-llm-evaluation-benchmarking-gemma-and-beyond-with-benchmark-gemma-models-9dc6a5266be8)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-green)](https://www.python.org/)
![Coverage](img/coverage.svg)
![Tests](https://img.shields.io/badge/tests-99%25_passed-blue.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![Powered by Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)](https://huggingface.co/transformers)
[![Uses PyTorch](https://img.shields.io/badge/Uses-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Uses TensorFlow](https://img.shields.io/badge/Uses-TensorFlow-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
## Unleash Your LLM Benchmarking: Extreme Flexibility & Resource Mastery

This framework is engineered for comprehensive, deeply customizable, and resource-conscious evaluation of Large Language Models (LLMs), with a primary focus on **Gemma models** and compatibility with **LLaMA**, **Mistral**, and other architectures. What sets this suite apart is its **unparalleled flexibility** through custom script integration for every core component (models, datasets, tasks, metrics, post-processing) and its meticulous **attention to resource optimization**, enabling extensive benchmarks even on constrained hardware.

Whether you're a researcher experimenting with novel evaluation techniques, a developer fine-tuning models, or an enthusiast comparing the latest LLMs, this suite provides a robust, automated, and scalable solution. Go beyond standard benchmarks by defining your own logic, seamlessly integrate models from diverse sources including PyTorch, TensorFlow, and custom environments, and gain granular insights with advanced logging and intelligent output processing, all while maintaining efficient memory and disk utilization.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Examples & Getting Started](#examples--getting-started)
  - [Quick Test with Colab](#quick-test-with-colab)
  - [Generating a Default Configuration](#generating-a-default-configuration)
  - [Visualizing Benchmark Results](#visualizing-benchmark-results)
- [Configuring Benchmarks (`benchmark_config.yaml`)](#configuring-benchmarks-benchmark_configyaml)
  - [Leveraging Custom Scripts for Ultimate Flexibility](#leveraging-custom-scripts-for-ultimate-flexibility)
- [Core Components](#core-components)
- [Key Advantages & Why Use This Benchmark Suite?](#key-advantages--why-use-this-benchmark-suite)
- [Testing](#testing)
- [Next Steps](#next-steps)
- [License and Important Disclaimers](#license-and-important-disclaimers)

## Overview

This framework offers a robust, automated, and scalable solution to assess the performance of Large Language Models (LLMs), with a primary focus on **Gemma models**. It provides insights across diverse tasks, including standard academic benchmarks and custom datasets. The system emphasizes scalability through efficient batch processing, stateful metrics, and intelligent resource management; extensibility via a modular factory-based architecture that now fully supports custom user scripts for all core components; and ease of use, ensuring researchers can efficiently measure, compare, and reproduce model performance with unprecedented control.

**Want a deeper dive into the philosophy, features, and how to get the most out of Benchmark-Gemma-Models?** Check out **my comprehensive blog post** on Medium:
**➡️ [Customizable LLM Evaluation: Benchmarking Gemma and Beyond with Benchmark-Gemma-Models](https://medium.com/@domenicolacavalla8/customizable-llm-evaluation-benchmarking-gemma-and-beyond-with-benchmark-gemma-models-9dc6a5266be8)**

---

## Directory Structure
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

---

## Core Components

| Component                      | Description                                                                                                                                                                                             |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `run_benchmark.py`             | Main script to execute benchmarks based on a configuration.                                                                                                                                             |
| `benchmark_config.yaml`        | Central YAML (Pydantic-validated) defining experiments: models, tasks, datasets, metrics, and settings.                                                                                                   |
| `BenchmarkRunner`              | Orchestrates the entire benchmark lifecycle: loading, execution, evaluation, and reporting.                                                                                                               |
| **Model Loading** | Manages LLM loading (HF, PyTorch, TF, Custom Scripts) with quantization/offloading. Governed by `ModelLoaderFactory`.                                                                                       |
| **Dataset Loading** | Handles data fetching (HF Hub, local, Custom Scripts), normalization, and streaming. Governed by `DatasetFactory`.                                                                                        |
| **Task Handling** | Manages task-specific logic, model interaction, and batch processing. Governed by `TaskHandlerFactory`.                                                                                                   |
| **Prompt Building** | Constructs model input prompts based on task configuration and templates. Governed by `PromptBuilderFactory`.                                                                                              |
| **Post-processing** | Cleans and structures raw model outputs for evaluation (e.g., using regex or custom logic). Governed by `PostProcessorFactory`.                                                                          |
| **Metric Evaluation** | Implements various evaluation measures (built-in or custom). The `Evaluator` manages their stateful, batch-by-batch computation, using `MetricFactory`.                                                      |
| `custom_loader_scripts/`       | Directory for user-defined Python scripts, enabling custom logic for models, datasets, tasks, post-processors, and metrics.                                                                                 |
| `FileManager`                  | Saves aggregated benchmark results in specified formats (JSON, CSV, PDF).                                                                                                                                 |
| `visualize/dashboard.py`       | Interactive Streamlit application for visualizing benchmark results.                                                                                                                                    |

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

## Testing

Basic testing has been implemented for this project using `pytest`. The current test suite focuses on covering the main functional areas (e.g., model loading, text generation) to ensure key components behave as expected.

> **Note:** Testing coverage is still limited and primarily targets high-level workflows rather than edge cases or detailed unit-level behavior.

### Running Tests

To run the existing tests, make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Then execute the tests from the root directory:
```bash
pytest
```

This command will automatically discover and run tests within the `tests/ directory`. For more advanced usage and command-line options (e.g., running specific tests, verbose output, generating reports), please refer to the [official pytest documentation](https://docs.pytest.org/en/latest/how-to/usage.html).

---

## Next Steps

We aim to strengthen the framework’s stability, usability, and reach, while welcoming community contributions and feedback.

### Immediate & Medium-Term Goals

* **Framework Solidification & Expansion:**
    * Improve testing and robustness by adding comprehensive unit and integration tests, especially for custom script functionalities.
    * Expand the suite of built-in evaluation metrics and streamline the integration of new academic benchmarks.
    * Enhance existing documentation (README, code comments) and create richer examples for custom scripts to improve clarity and ease of use.
    * Refine GLUE task support, focusing on prompt templates and output parsing for more reliable results.
* **Community Engagement & Initial Promotion:**
    * Share the project through blog posts and social media (e.g., LinkedIn) to gather initial user feedback.
    * Actively address reported issues and incorporate valuable suggestions from the community.

### Future Vision & Enhancements

* **Comprehensive Documentation & Advanced Features:**
    * Develop a dedicated GitHub Pages documentation site for in-depth guides and tutorials if community interest and project complexity warrant it.
    * Enhance the Streamlit dashboard with advanced visualization and error analysis capabilities.
    * Explore options for simplified cloud execution (Colab Pro, AWS, GCP) and potential database integration for longitudinal result tracking.
* **Broader Ecosystem Support:**
    * Continuously add compatibility for new LLMs, model serving frameworks, and evolving evaluation methodologies based on community demand and advancements in the field.

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