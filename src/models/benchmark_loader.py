import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from evaluation.evaluator import Evaluator
from models.models_factory import ModelLoaderFactory
from utils.file_manager import save_results
from utils.logger import setup_logger
from datasets import load_dataset

logger = setup_logger()

class BenchmarkRunner:
    """
    Class to run benchmarking tasks across multiple models and datasets using configurations.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BenchmarkRunner with the provided configuration.

        :param config: Dictionary containing configuration for models, tasks, metrics, etc.
        """
        self.config = config
        self.models = config["models"]
        self.tasks = config["tasks"]
        self.model_params = config.get("model_parameters", {})
        self.evaluation_params = config.get("evaluation", {})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}
    
    def run(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run the benchmark across all models and tasks.

        :return: A dictionary containing the results for each model/task combination.
        """
        for model_config in self.models:
            model_name = model_config["name"]
            framework = model_config["framework"]
            quantization = model_config.get("quantization")  # Get quantization parameter
            logger.info(f"Running benchmark for model: {model_name} using framework: {framework} with quantization: {quantization}")
            
            # Use the factory to get the appropriate model loader
            model_loader = ModelLoaderFactory.get_model_loader(
                model_name=model_name,
                framework=framework,
                quantization=quantization,
                **self.model_params
            )
            model, tokenizer = model_loader.load(quantization=quantization)

            for task_config in self.tasks:
                task_name = task_config["name"]
                logger.info(f"Running task {task_name} for model {model_name}...")

                # Extract task-specific evaluation metrics
                task_metrics = task_config["evaluation_metrics"]

                # Run the task and obtain predictions and labels
                predictions, labels = self._run_task(model, tokenizer, task_config)

                # Create an Evaluator instance with evaluation parameters
                evaluator = Evaluator(self.evaluation_params)

                # Prepare task results dictionary
                task_results = {"predictions": predictions, "labels": labels}

                # Ensure task_metrics is a list of dictionaries
                formatted_task_metrics = [{"name": metric} if isinstance(metric, str) else metric for metric in task_metrics]

                # Compute the metrics
                evaluation_results = evaluator.evaluate(task_results, formatted_task_metrics)

                # Store the results
                if model_name not in self.results:
                    self.results[model_name] = {}
                self.results[model_name][task_name] = evaluation_results

        # Save the results to a file
        save_results(self.results, "benchmark_results.json")
        return self.results
    
    def _run_task(self, model: Any, tokenizer: Any, task_config: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """
        Run a specific task using the model.

        :param model: The loaded model.
        :param tokenizer: The tokenizer for the model.
        :param task_config: Configuration for the task to run (e.g., MMLU, GSM8K).
        :return: A tuple containing predictions and labels.
        """
        task_name = task_config["name"]
        dataset_name = task_config["datasets"][0]["name"]
        # Get the dataset configuration (e.g., 'all', 'abstract_algebra', etc.)
        dataset_config = task_config["datasets"][0].get("config", "all")  # Default to "all" if no config is provided
        dataset_split = task_config["datasets"][0]["splits"][0]  # Assume we use the 'train' or 'validation' split

        logger.info(f"Loading dataset {dataset_name} with config {dataset_config} ({dataset_split})...")
        # Load the dataset with the specified configuration
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)

        predictions = []
        labels = []

        # Iterate through the dataset
        for example in dataset:
            
            input_text = example.get("text", "")  # Use .get() to avoid KeyError if 'text' is missing
            label = example.get("label", None)  # Adjust the key as needed based on dataset
            if label is not None:
                labels.append(label)

            # Tokenize the input text
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(self.device)

            if task_config["type"] == "classification":
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    predictions.append(predicted_class)

            elif task_config["type"] == "generation":
                with torch.no_grad():
                    generated_tokens = model.generate(**inputs, max_length=100)
                    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                    predictions.append(generated_text)

        return predictions, labels