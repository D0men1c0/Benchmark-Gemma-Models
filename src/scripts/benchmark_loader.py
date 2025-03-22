import torch
from typing import Dict, List, Any, Tuple
from datasets import Dataset, IterableDataset
import tqdm
from datasets.dataset_factory import DatasetFactory
from .evaluator import Evaluator
from models.models_factory import ModelLoaderFactory
from tasks.task_handlers_factory import TaskHandlerFactory
from utils.file_manager import save_results
from utils.logger import setup_logger

logger = setup_logger()

class BenchmarkRunner:
    """
    Class to run benchmarking tasks across multiple models and datasets using configurations.
    """

    def __init__(self, config: Dict[str, Any]):
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

    # Usage in BenchmarkRunner
    def _run_task(
        self, model: Any, tokenizer: Any, task_config: Dict[str, Any]
    ) -> Tuple[List[Any], List[Any]]:
        
        # Load dataset configuration
        dataset_config = task_config["datasets"][0]
        loader = DatasetFactory.from_config(dataset_config)
        
        try:
            dataset = loader.load()
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
        
        # Handle both regular and iterable datasets
        if isinstance(dataset, (Dataset, IterableDataset)):
            data_iter = dataset.iter(batch_size=1) if loader.streaming else dataset
        else:
            data_iter = dataset  # Custom iterables

        handler = TaskHandlerFactory.get_handler(
            task_config["type"], model, tokenizer, self.device
        )

        predictions = []
        labels = []

        for batch in tqdm(data_iter, desc="Processing examples"):
            # Handle batch unpacking for streaming
            if loader.streaming:
                example = {k: v[0] for k, v in batch.items()}  # Unpack first item in batch
            else:
                example = batch
                
            prediction, label = handler.process_example(example)
            predictions.append(prediction)
            if label is not None:
                labels.append(label)

        return predictions, labels