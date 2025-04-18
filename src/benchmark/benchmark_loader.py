import logging
import torch
from typing import Dict, List, Any, Tuple, Iterable
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config_models import BenchmarkConfig, TaskConfig
from .dataset.dataset_factory import DatasetFactory
from .evaluation.evaluator import Evaluator
from .models.models_factory import ModelLoaderFactory
from .tasks.task_handlers_factory import TaskHandlerFactory
from .reporting.file_manager import save_results
from utils.logger import setup_logger

class BenchmarkRunner:
    """
    Class to run benchmarking tasks across multiple models and datasets using configurations.
    """

    def __init__(self, config: BenchmarkConfig):
        self.logger = setup_logger(self.__class__.__name__)
        self.config: BenchmarkConfig = config
        self.output_dir = config.general.output_dir
        self.reporting_format = config.reporting.format
        self.device = self.determine_device()
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def determine_device(self) -> str:
        """Determines the device to use (CUDA, MPS, CPU)."""
        if torch.cuda.is_available():
            self.logger.info("CUDA is available. Using GPU.")
            return "cuda"
        elif torch.backends.mps.is_available():
             self.logger.info("MPS is available. Using Apple Silicon GPU.")
             return "mps"
        else:
            self.logger.info("CUDA and MPS not available. Using CPU.")
            return "cpu"

    def run(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run the benchmark across all models and tasks. Optimizes dataset loading.

        :return: A dictionary containing the results for each model/task combination.
        """
        task_datasets = {}
        self.logger.info("Pre-loading datasets for all tasks...")
        if not hasattr(self.config, 'tasks') or not self.config.tasks:
             self.logger.error("No tasks defined in the configuration. Aborting.")
             return {}

        for task_config in self.config.tasks:
            task_name = task_config.name
            try:
                if not hasattr(task_config, 'datasets') or not task_config.datasets:
                    self.logger.warning(f"Task '{task_name}' has no datasets defined. Skipping dataset loading for this task.")
                    continue

                dataset_config = task_config.datasets[0]
                self.logger.info(f"Loading dataset '{dataset_config.name}' for task '{task_name}'...")
                loader = DatasetFactory.from_config(dataset_config.dict(exclude_none=True))
                dataset = loader.load() # Load or get the iterable
                task_datasets[task_name] = {
                    "dataset": dataset,
                    "streaming": loader.streaming # Store streaming info
                }
                self.logger.info(f"Dataset '{dataset_config.name}' ready for task '{task_name}'.")
            except Exception as e:
                self.logger.error(f"Failed to load dataset for task {task_name}: {e}. This task will be skipped for all models.", exc_info=True)
                # Decide whether to continue with other tasks or stop everything
                task_datasets[task_name] = None # Mark as failed

        self.logger.info("Starting benchmark runs for each model...")
         # Check if models are defined
        if not hasattr(self.config, 'models') or not self.config.models:
            self.logger.error("No models defined in the configuration. Aborting.")
            return {}

        # Loop over models
        for model_config in self.config.models:
            model_name = model_config.name
            framework = model_config.framework
            quantization = model_config.quantization

            self.logger.info(f"--- Processing Model: {model_name} (Framework: {framework}, Quantization: {quantization}) ---")

            model = None
            tokenizer = None
            try:
                # Load model and tokenizer ONCE per model
                # Combine model_parameters and potential model_config specific ones
                model_load_params = self.config.model_parameters.dict(exclude_none=True)

                model_loader = ModelLoaderFactory.get_model_loader(
                    model_name=model_config.checkpoint or model_name, # Use checkpoint if available
                    framework=framework,
                    quantization=quantization,
                    **model_load_params # Pass relevant model parameters
                )
                self.logger.info(f"Loading model '{model_name}' and tokenizer...")
                model, tokenizer = model_loader.load(quantization=quantization)

                if hasattr(model, 'to'):
                     model.to(self.device)
                else:
                     self.logger.warning(f"Model object of type {type(model)} might not support '.to()' method for device placement.")

                self.logger.info(f"Model '{model_name}' loaded successfully on device '{self.device}'.")

            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}. Skipping this model.", exc_info=True)
                # Free memory if loading partially failed
                del model
                del tokenizer
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                     try:
                          import torch.mps
                          torch.mps.empty_cache()
                     except ImportError:
                           pass # MPS might not be available
                gc.collect()
                continue # Skip to the next model

            # Loop over tasks for the current model
            for task_config in self.config.tasks:
                task_name = task_config.name

                # Check if the dataset for this task was loaded correctly
                if task_name not in task_datasets or task_datasets[task_name] is None:
                    self.logger.warning(f"Skipping task '{task_name}' for model '{model_name}' because dataset loading failed earlier.")
                    continue

                self.logger.info(f"Running task '{task_name}' for model '{model_name}'...")
                dataset_info = task_datasets[task_name]
                dataset = dataset_info["dataset"]
                is_streaming = dataset_info["streaming"]

                try:
                    # Run the task using model/tokenizer and pre-loaded dataset
                    predictions, labels = self._run_task(
                        model, tokenizer, task_config, dataset, is_streaming
                    )

                    if not predictions and not labels:
                         self.logger.warning(f"Task '{task_name}' for model '{model_name}' produced no results. Skipping evaluation.")
                         continue # Skip evaluation if no results

                    eval_params = self.config.evaluation or {}
                    evaluator = Evaluator(eval_params)
                    task_results_dict = {"predictions": predictions, "labels": labels}

                    formatted_task_metrics = [m.dict(exclude_none=True) for m in task_config.evaluation_metrics]

                    evaluation_results = evaluator.evaluate(task_results_dict, formatted_task_metrics)

                    # Store results
                    if model_name not in self.results:
                        self.results[model_name] = {}
                    self.results[model_name][task_name] = evaluation_results
                    self.logger.info(f"Task '{task_name}' completed for model '{model_name}'. Results summary: {list(evaluation_results.keys())}")
                    if self.logger.isEnabledFor(logging.DEBUG):
                         self.logger.debug(f"Detailed results for task '{task_name}', model '{model_name}': {evaluation_results}")


                except Exception as e:
                    self.logger.error(f"Failed to run or evaluate task '{task_name}' for model '{model_name}': {e}", exc_info=True)
                    # Record failure but continue with the next task/model
                    if model_name not in self.results:
                        self.results[model_name] = {}
                    self.results[model_name][task_name] = {"error": str(e)}

            # --- Optimization: GPU/MPS Memory Cleanup after processing ALL tasks for a model ---
            self.logger.info(f"Finished all tasks for model {model_name}. Cleaning up resources...")
            del model
            del tokenizer
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache.")
            elif self.device == 'mps':
                try:
                    import torch.mps
                    torch.mps.empty_cache()
                    self.logger.info("Cleared MPS cache.")
                except ImportError:
                    pass # MPS might not be available
            gc.collect() # Force garbage collection
            self.logger.info(f"Resources cleaned up for model {model_name}.")
            # --- End Memory Cleanup ---

        # Save final results
        if not self.results:
             self.logger.warning("No benchmark results were generated.")
        else:
            try:
                # Ensure output_dir is created
                self.output_dir.mkdir(parents=True, exist_ok=True)
                save_results(
                    results=self.results,
                    output_dir=str(self.output_dir), # save_results expects string
                    format=self.reporting_format
                )
            except Exception as e:
                 self.logger.error(f"Failed to save benchmark results: {e}", exc_info=True)

        return self.results

    def _run_task(
        self,
        model: Any,
        tokenizer: Any,
        task_config: TaskConfig,
        dataset: Iterable,
        is_streaming: bool
    ) -> Tuple[List[Any], List[Any]]:
        """
        Run a single task using the provided model, tokenizer, and pre-loaded dataset.

        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param task_config: Task configuration object.
        :param dataset: Pre-loaded dataset iterable (can be Dataset, IterableDataset, or custom).
        :param is_streaming: Boolean indicating if the dataset is streaming.
        :return: Tuple of predictions and labels.
        """
        task_type = task_config.type
        task_name = task_config.name

        # Get the handler (pass config.advanced for parameters)
        adv_conf_dict = self.config.advanced.dict(exclude_none=True) if self.config.advanced else {}
        handler = TaskHandlerFactory.get_handler(
            task_type, model, tokenizer, self.device, adv_conf_dict
        )

        predictions = []
        labels = []

        # Use DataLoader for efficient batch processing
        # Note: batch_size taken from config.advanced
        batch_size = self.config.advanced.batch_size if self.config.advanced else 32 # Default if advanced missing
        try:
            # DataLoader handles both map-style (Dataset) and iterable-style (IterableDataset)
            # shuffle=False is important for reproducibility and result ordering
            # Disable persistent_workers and num_workers for basic iterable datasets if issues arise
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0, # Start with 0, increase if I/O bound and dataset allows
                pin_memory= (self.device == 'cuda') # Pin memory only if using CUDA
            )
        except Exception as e:
            self.logger.error(f"Failed to create DataLoader for task '{task_name}': {e}. Returning empty results.", exc_info=True)
            return [], []

        # Determine total number of batches if possible (for tqdm)
        total_batches = None
        # Check if dataset has __len__ and is NOT streaming (streaming datasets usually don't have a fixed len)
        if not is_streaming and hasattr(dataset, '__len__'):
             try:
                 dataset_len = len(dataset) # type: ignore [arg-type]
                 total_batches = (dataset_len + batch_size - 1) // batch_size
             except TypeError:
                  self.logger.warning(f"Dataset for task '{task_name}' has __len__ but it raised TypeError. Progress bar total might be inaccurate.")

        self.logger.info(f"Processing task '{task_name}' with batch size {batch_size}...")
        # Use tqdm for progress bar
        progress_bar = tqdm(data_loader, desc=f"Processing task '{task_name}'", total=total_batches, unit="batch", leave=False)

        batch_num = 0
        try:
            for batch in progress_bar:
                batch_num += 1
                try:
                    # The handler expects a dictionary of lists (the batch from DataLoader)
                    predictions_batch, labels_batch = handler.process_batch(batch)

                    # Handle case where process_batch might not return valid lists
                    if predictions_batch is not None:
                        predictions.extend(predictions_batch)
                    if labels_batch is not None:
                        labels.extend(labels_batch)

                except Exception as e:
                    # Log error for the specific batch, decide whether to continue
                    self.logger.error(f"Error processing batch {batch_num} for task '{task_name}': {e}", exc_info=True)
                    # Option: continue to next batch (might lead to incomplete/misaligned results)
                    # Option: break the loop for this task (safer)
                    self.logger.warning(f"Stopping processing for task '{task_name}' due to error in batch {batch_num}.")
                    break # Stop processing this task

        except Exception as e:
             # Catch potential errors during DataLoader iteration itself
             self.logger.error(f"DataLoader iteration failed for task '{task_name}' after {batch_num} batches: {e}", exc_info=True)
        finally:
             progress_bar.close() # Ensure progress bar is closed

        if not predictions and not labels:
             self.logger.warning(f"No predictions or labels were generated for task '{task_name}'. Check dataset and processing logic.")

        return predictions, labels