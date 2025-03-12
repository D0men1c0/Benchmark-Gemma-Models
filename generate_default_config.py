import yaml
import os
import logging

def generate_default_config(config_path: str = "src/config/benchmark_config.yaml") -> None:
    """
    Creates a YAML configuration file. If the file already exists,
    saves a new version with an incrementing suffix (e.g., benchmark_config(1).yaml).

    :param config_path: Desired path for the configuration file.
    """
    base_name, ext = os.path.splitext(config_path)
    file_path = config_path
    counter = 1

    # Check for existing file and create incremented versions
    while os.path.exists(file_path):
        file_path = f"{base_name}({counter}){ext}"
        counter += 1

    default_config = {
        "general": {
            "experiment_name": "Gemma_Benchmark_2025",
            "output_dir": "./benchmarks",
            "random_seed": 42
        },
        "tasks": [
            {
                "name": "MMLU",
                "type": "classification",
                "description": "Massive Multitask Language Understanding",
                "datasets": [{"name": "mmlu", "type": "classification", "splits": ["train", "validation"]}],
                "evaluation_metrics": ["accuracy", "f1_score", "perplexity"]
            },
            {
                "name": "GSM8K",
                "type": "generation",
                "description": "Math dataset with problem-solving tasks",
                "datasets": [{"name": "gsm8k", "type": "generation", "splits": ["train", "validation"]}],
                "evaluation_metrics": ["accuracy", "execution_time"]
            }
        ],
        "models": [
            {"name": "gemma-7b", "variant": "gemma", "size": "7B", "framework": "huggingface", "checkpoint": "gemma-7b-checkpoint", "quantization": "4bit", "offloading": True},
            {"name": "gemma-13b", "variant": "gemma", "size": "13B", "framework": "huggingface", "checkpoint": "gemma-13b-checkpoint", "quantization": "8bit", "offloading": True}
        ],
        "model_parameters": {
            "batch_size": 4,
            "max_input_length": 512,
            "max_output_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        },
        "evaluation": {
            "batch_size": 16,
            "log_interval": 100
        },
        "reporting": {
            "enabled": True,
            "format": "pdf",
            "leaderboard_enabled": True,
            "generate_visuals": {"charts": True, "tables": True, "save_plots": True},
            "output_dir": "./reports"
        },
        "advanced": {
            "enable_multi_gpu": False,
            "use_tpu": False,
            "distributed_training": False
        }
    }

    with open(file_path, "w") as file:
        yaml.dump(default_config, file, default_flow_style=False)

    logging.info(f"Default configuration file created at {file_path}.")

if __name__ == "__main__":
    generate_default_config()