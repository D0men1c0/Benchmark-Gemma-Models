import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Any

from config_models import (
    BenchmarkConfig,
    GeneralConfig,
    DatasetConfig,
    MetricConfig,
    TaskConfig,
    ModelConfig,
    ModelParamsConfig,
    EvaluationConfig,
    ReportingConfig,
    AdvancedConfig
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DefaultConfigProvider:
    """Provides default configurations for the benchmark pipeline."""

    def __init__(self) -> None:
        """Initialize the DefaultConfigProvider with a timestamp for unique naming."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_default_general_config(self) -> GeneralConfig:
        """Generate a default general configuration."""
        return GeneralConfig(
            experiment_name=f"Gemma_Benchmark_Comprehensive_{self.timestamp}",
            output_dir=Path("benchmarks_output"),
            random_seed=42
        )

    def get_default_mmlu_task(self) -> TaskConfig:
        """Generate a default task configuration for MMLU (Massive Multitask Language Understanding)."""
        prompt_template = (
            "The following is a multiple-choice question about {subject}.\n"
            "Please choose the letter of the correct answer.\n"
            "Question: {question}\nChoices:\n{choices_formatted_str}\nAnswer:"
        )
        return TaskConfig(
            name="MMLU (all Subset - Templated)",
            type="multiple_choice_qa",
            description="MMLU (all subjects) via templated letter generation.",
            datasets=[
                DatasetConfig(
                    name="cais/mmlu", source_type="hf_hub", config="all",
                    split="validation", max_samples=100
                )
            ],
            handler_options={
                "prompt_builder_type": "mmlu",
                "prompt_template": prompt_template,
                "default_subject": "the following topic"
            },
            evaluation_metrics=[
                MetricConfig(name="exact_match", options={"normalize": True, "ignore_case": True, "ignore_punct": True}),
                MetricConfig(name="accuracy", options={}),
                MetricConfig(name="f1_score", options={"average": "weighted"}),
                MetricConfig(name="precision", options={"average": "weighted"}),
                MetricConfig(name="recall", options={"average": "weighted"}),
                MetricConfig(name="rouge", options={"metrics": ['rouge1', 'rouge2', 'rougeL'], "stats": ['f']}),
                MetricConfig(name="bleu", options={}),
                MetricConfig(name="meteor", options={}),
                MetricConfig(name="bert_score", options={"lang": "en"})
            ]
        )

    def get_default_gsm8k_task(self) -> TaskConfig:
        """Generate a default task configuration for GSM8K (Grade School Math 8K)."""
        prompt_template = (
            "Solve the following math problem step-by-step. "
            "Your final answer should be a number, formatted as ####<number>.\n\n"
            "Problem: {input_text}\n\nSolution:"
        )
        return TaskConfig(
            name="GSM8K (Reasoning Text Analysis)",
            type="math_reasoning_generation",
            description="Grade school math problems with step-by-step reasoning and final numeric answer.",
            datasets=[
                DatasetConfig(name="gsm8k", source_type="hf_hub", config="main", split="test", max_samples=100)
            ],
            handler_options={
                "prompt_builder_type": "default",
                "prompt_template": prompt_template,
                "postprocessor_key": "gsm8k"
            },
            evaluation_metrics=[
                MetricConfig(name="exact_match", options={}),
                MetricConfig(name="rouge", options={"metrics": ['rouge1', 'rouge2', 'rougeL'], "stats": ['f', 'p', 'r']}),
                MetricConfig(name="bert_score", options={"lang": "en"}),
                MetricConfig(name="distinct_ngram", options={"ngrams": [1, 2, 3]}),
                MetricConfig(name="word_entropy", options={})
            ]
        )

    def get_default_summarization_task(self) -> TaskConfig:
        """Generate a default task configuration for summarization."""
        prompt_template = (
            "Summarize the following article in a few sentences:\n\n"
            "Article: {article_text}\n\nSummary:"
        )
        return TaskConfig(
            name="CNN/DailyMail Summarization",
            type="summarization",
            description="Abstractive summarization of news articles.",
            datasets=[
                DatasetConfig(name="cnn_dailymail", source_type="hf_hub", config="3.0.0", split="test", max_samples=100)
            ],
            handler_options={
                "prompt_builder_type": "default",
                "prompt_template": prompt_template,
                "postprocessor_key": "summarization"
            },
            evaluation_metrics=[
                MetricConfig(name="rouge", options={"metrics": ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], "stats": ['f', 'p', 'r']}),
                MetricConfig(name="bert_score", options={"lang": "en"}),
                MetricConfig(name="meteor", options={}),
                MetricConfig(name="bleu", options={}),
                MetricConfig(name="distinct_ngram", options={"ngrams": [1, 2, 3, 4]}),
                MetricConfig(name="word_entropy", options={})
            ]
        )

    def get_default_translation_task(self) -> TaskConfig:
        """Generate a default task configuration for translation."""
        prompt_template = "Translate the following {source_lang_name} text into {target_lang_name}: {input_text}"
        return TaskConfig(
            name="OPUS-100 English-to-French",
            type="translation",
            description="Translate sentences from English to French using OPUS-100.",
            datasets=[
                DatasetConfig(name="opus100", source_type="hf_hub", config="en-fr", split="validation", max_samples=100)
            ],
            handler_options={
                "prompt_builder_type": "translation",
                "prompt_template": prompt_template,
                "postprocessor_key": "translation"
            },
            evaluation_metrics=[
                MetricConfig(name="bleu", options={}),
                MetricConfig(name="meteor", options={}),
                MetricConfig(name="rouge", options={"metrics": ['rougeL'], "stats": ['f']}),
                MetricConfig(name="bert_score", options={"lang": "fr"})
            ]
        )

    def get_default_models(self) -> List[ModelConfig]:
        """Generate a list of default model configurations."""
        return [
            ModelConfig(
                name="gemma-7b", variant="gemma", size="7B", framework="huggingface",
                checkpoint="google/gemma-7b", quantization="4bit", offloading=True, torch_dtype="bfloat16"
            ),
            ModelConfig(
                name="gemma-2b", variant="gemma", size="2B", framework="huggingface",
                checkpoint="google/gemma-2b", quantization="4bit", offloading=True, torch_dtype="bfloat16"
            ),
            ModelConfig(
                name="gpt2", variant="gpt2", size="124M", framework="huggingface",
                checkpoint="gpt2", quantization=None, offloading=False, torch_dtype="float32"
            )
        ]

    def get_default_model_params_config(self) -> ModelParamsConfig:
        """Generate a default model parameters configuration."""
        return ModelParamsConfig(temperature=None, top_p=None, top_k=None)

    def get_default_evaluation_config(self) -> EvaluationConfig:
        """Generate a default evaluation configuration."""
        return EvaluationConfig(log_interval=5)

    def get_default_reporting_config(self) -> ReportingConfig:
        """Generate a default reporting configuration."""
        return ReportingConfig(
            enabled=True,
            format="json",
            leaderboard_enabled=False,
            generate_visuals={"charts": True, "tables": True, "save_plots": True},
            output_dir=Path("reports")
        )

    def get_default_advanced_config(self) -> AdvancedConfig:
        """Generate a default advanced configuration."""
        return AdvancedConfig(
            enable_multi_gpu=False, use_tpu=False, distributed_training=False,
            batch_size=10, truncation=True, padding=True,
            generate_max_length=512, skip_special_tokens=True, max_new_tokens=150,
            num_beams=1, do_sample=False, use_cache=True, clean_up_tokenization_spaces=True
        )

    def get_default_benchmark_config(self) -> BenchmarkConfig:
        """Generate a complete default benchmark configuration."""
        return BenchmarkConfig(
            general=self.get_default_general_config(),
            tasks=[
                self.get_default_mmlu_task(),
                self.get_default_gsm8k_task(),
                self.get_default_summarization_task(),
                self.get_default_translation_task()
            ],
            models=self.get_default_models(),
            model_parameters=self.get_default_model_params_config(),
            evaluation=self.get_default_evaluation_config(),
            reporting=self.get_default_reporting_config(),
            advanced=self.get_default_advanced_config()
        )


class ConfigGenerator:
    """Generates and writes benchmark configuration YAML files."""

    def __init__(self, provider: DefaultConfigProvider) -> None:
        """
        Initialize the ConfigGenerator with a DefaultConfigProvider.
        :param provider: An instance of DefaultConfigProvider.
        """
        self.provider = provider
        self.timestamp = provider.timestamp

    def generate_and_save(self, output_dir_str: str = "src/config", base_filename: str = "default_benchmark_config") -> None:
        """
        Generate and save a benchmark configuration YAML file.

        :param output_dir_str: Directory where the file will be saved.
        :param base_filename: Base name of the configuration file.
        """
        benchmark_config_instance = self.provider.get_default_benchmark_config()
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{base_filename}.yaml"
        output_path = output_dir / file_name

        yaml_output_parts: List[str] = []

        def append_yaml_section(data_object: Any, comment_lines: List[str]) -> None:
            if data_object is None:
                return

            for comment in comment_lines:
                yaml_output_parts.append(f"# {comment}\n")

            dumpable_data: Any
            if isinstance(data_object, list):
                dumpable_data = [item.model_dump(mode="json", exclude_none=True) for item in data_object]
            else:
                dumpable_data = data_object.model_dump(mode="json", exclude_none=True)

            key_name = ""
            for field_name, field_info in BenchmarkConfig.model_fields.items():
                is_list_field = hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ in (list, List)
                if is_list_field:
                    item_type = field_info.annotation.__args__[0]
                    if isinstance(data_object, list) and data_object and isinstance(data_object[0], item_type):
                        key_name = field_name
                        break
                elif isinstance(data_object, field_info.annotation):
                    key_name = field_name
                    break

            if not key_name:
                logging.warning(f"Could not determine YAML key for object: {type(data_object)}. Skipping.")
                return

            section_yaml = yaml.dump(
                {key_name: dumpable_data},
                sort_keys=False,
                allow_unicode=True,
                width=1000,
                default_flow_style=False,
                indent=2
            )
            yaml_output_parts.append(section_yaml + "\n")

        append_yaml_section(benchmark_config_instance.general, ["============================", "General Configuration", "============================"])
        append_yaml_section(benchmark_config_instance.tasks, ["============================", "Benchmark Tasks Configuration", "============================"])
        append_yaml_section(benchmark_config_instance.models, ["============================", "Models Configuration", "============================"])

        yaml_output_parts.append("# ======================================================================\n")
        yaml_output_parts.append("# Model Parameters Configuration (Global for model loading, not generation)\n")
        yaml_output_parts.append("# ======================================================================\n")
        yaml_output_parts.append(yaml.dump(
            {"model_parameters": {"max_input_length": 512, "max_output_length": 512}},
            sort_keys=False, allow_unicode=True, width=1000, default_flow_style=False, indent=2
        ) + "\n")

        append_yaml_section(benchmark_config_instance.evaluation, ["============================", "Evaluation Configuration", "============================"])
        append_yaml_section(benchmark_config_instance.reporting, ["============================", "Results and Reporting", "============================"])
        append_yaml_section(benchmark_config_instance.advanced, ["============================", "Advanced Settings", "============================"])

        try:
            with output_path.open("w", encoding="utf-8") as f:
                f.writelines(yaml_output_parts)
            logging.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to write YAML configuration: {e}", exc_info=True)


if __name__ == "__main__":
    provider = DefaultConfigProvider()
    generator = ConfigGenerator(provider)
    generator.generate_and_save(output_dir_str="src/config", base_filename="default_benchmark_config")