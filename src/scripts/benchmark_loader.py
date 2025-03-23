import torch
from typing import Dict, List, Any, Tuple
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from dataset.dataset_factory import DatasetFactory
from torch.utils.data import DataLoader
from .evaluator import Evaluator
from models.models_factory import ModelLoaderFactory
from tasks.task_handlers_factory import TaskHandlerFactory
from utils.file_manager import save_results
from utils.logger import setup_logger

class BenchmarkRunner:
    """
    Class to run benchmarking tasks across multiple models and datasets using configurations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = setup_logger()
        self.config = config
        self.models = config.get("models", [])
        self.advanced = config.get("advanced", {})
        self.tasks = config.get("tasks", [])
        self.model_params = config.get("model_parameters", {})
        self.evaluation_params = config.get("evaluation", {})
        self.output_dir = self.config.get("general", {}).get("output_dir", "results")
        self.reporting_format = self.config.get("reporting", {}).get("format", "json")
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
            self.logger.info(f"Running benchmark for model: {model_name} using framework: {framework} with quantization: {quantization}")
            
            # Use the factory to get the appropriate model loader
            model_loader = ModelLoaderFactory.get_model_loader(
                model_name=model_name,
                framework=framework,
                quantization=quantization,
                **self.model_params
            )
            self.logger.info(f"Loaded model: {model_loader} using {framework} framework")
            model, tokenizer = model_loader.load(quantization=quantization)

            for task_config in self.tasks:
                task_name = task_config["name"]
                self.logger.info(f"Running task {task_name} for model {model_name}...")

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
        save_results(
            results=self.results,
            output_dir=self.output_dir,
            format=self.reporting_format
        )
        return self.results


    def _run_task(
        self, model: Any, tokenizer: Any, task_config: Dict[str, Any]
    ) -> Tuple[List[Any], List[Any]]:
        """"
        Run a task using the provided model and tokenizer."

        :param model: Model instance to use for the task.
        :param tokenizer: Tokenizer instance to use for the task.
        :param task_config: Task configuration dictionary.
        """
        
        # Load dataset configuration
        dataset_config = task_config["datasets"][0]
        loader = DatasetFactory.from_config(dataset_config)
        
        try:
            dataset = loader.load()
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise
        
        # Handle both regular and iterable datasets
        if isinstance(dataset, (Dataset, IterableDataset)):
            data_iter = dataset.iter(batch_size=1) if loader.streaming else dataset
        else:
            data_iter = dataset  # Custom iterables

        if isinstance(data_iter, IterableDataset):
            iterable_dataset = data_iter
        else:
            iterable_dataset = CustomIterableDataset(data_iter)

        # Initialize the task handler based on task type
        handler = TaskHandlerFactory.get_handler(
            task_config["type"], model, tokenizer, self.device, self.advanced
        )

        predictions = []
        labels = []

        # Use DataLoader for batch processing
        data_loader = DataLoader(iterable_dataset, batch_size=self.advanced.get("batch_size", 32), shuffle=True)

        for batch in tqdm(data_loader, desc="Processing examples", unit="batch"):
            if loader.streaming:
                example = {k: v[0] for k, v in batch.items()}  # Unpack first item in batch
            else:
                example = batch
            
            # Process a batch of examples
            predictions_batch, labels_batch = handler.process_batch(example)
            predictions.extend(predictions_batch)
            if labels_batch is not None:
                labels.extend(labels_batch)

        return predictions, labels
    

class CustomIterableDataset(IterableDataset):
    """Custom IterableDataset to handle custom data iterators."""

    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        return iter(self.data_iter)