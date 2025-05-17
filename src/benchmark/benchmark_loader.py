import logging
import torch
from typing import Dict, List, Any, Tuple, Iterable, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, IterableDataset
import gc
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_models import BenchmarkConfig, TaskConfig, ModelConfig
from .dataset.dataset_factory import DatasetFactory
from .evaluation.evaluator import Evaluator
from .models.models_factory import ModelLoaderFactory
from .tasks.task_handlers_factory import TaskHandlerFactory
from .reporting.file_manager import save_results
from utils.logger import setup_logger

class BenchmarkRunner:
    """
    Runs benchmarking tasks based on provided configuration.
    """
    def __init__(self, config: BenchmarkConfig):
        self.logger = setup_logger(self.__class__.__name__)
        self.config: BenchmarkConfig = config
        self.output_dir: Path = config.general.output_dir
        self.reporting_format: str = config.reporting.format
        self.device: str = self._determine_device()
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {} # Stores {model: {task: {metric: value}}}

    def _determine_device(self) -> str:
        """Determines and logs the compute device."""
        if torch.cuda.is_available():
            self.logger.info("CUDA available. Using GPU.")
            return "cuda"
        else:
            self.logger.info("No GPU detected. Using CPU.")
            return "cpu"

    def _load_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads datasets for all tasks defined in the config.
        :return: Dictionary of datasets structured as {task_name: {dataset: dataset_instance, streaming: bool}}
        """
        task_datasets = {}
        self.logger.info("Pre-loading datasets...")
        if not self.config.tasks:
             self.logger.error("No tasks defined in config. Aborting dataset loading.")
             return {}

        for task_cfg in self.config.tasks:
            task_name = task_cfg.name
            if not task_cfg.datasets:
                self.logger.warning(f"Task '{task_name}' has no datasets defined. Skipping.")
                continue
            try:
                # Assuming one dataset per task for simplicity in this structure
                ds_cfg = task_cfg.datasets[0]
                self.logger.debug(f"Loading dataset '{ds_cfg.name}' for task '{task_name}' (type: {task_cfg.type})...")
                loader = DatasetFactory.from_config(ds_cfg.dict(exclude_none=True))
                dataset = loader.load(task_type=task_cfg.type) # Pass task_type for normalization
                task_datasets[task_name] = {
                    "dataset": dataset,
                    "streaming": loader.streaming,
                    "task_config": task_cfg # Keep task config for later use
                }
                self.logger.debug(f"Dataset '{ds_cfg.name}' ready for task '{task_name}'.")
            except Exception as e:
                self.logger.error(f"Failed to load dataset for task '{task_name}': {e}. This task will be skipped.", exc_info=True)
                task_datasets[task_name] = None # Mark as failed
        self.logger.info("Dataset loading complete.")
        return task_datasets

    def _load_model_and_tokenizer(self, model_cfg: ModelConfig) -> Optional[Tuple[Any, Any]]:
        """
        Loads the model and tokenizer based on the provided configuration.
        :param model_cfg: Configuration for the model to be loaded.
        :return: Tuple of (model, tokenizer) if successful, None otherwise.
        """
        model_name = model_cfg.name
        framework = model_cfg.framework
        quantization = model_cfg.quantization
        self.logger.info(f"--- Loading Model: {model_name} ({framework}, Quant: {quantization}) ---")
        try:
            model_load_params = self.config.model_parameters.dict(exclude_none=True)
            model_loader = ModelLoaderFactory.get_model_loader(
                model_name=model_cfg.checkpoint or model_name,
                framework=framework,
                quantization=quantization, 
                **model_load_params
            )
            model, tokenizer = model_loader.load(quantization=quantization)

            device_placement_handled_by_loader = False
            if model_cfg.quantization in ["4bit", "8bit"]: # Check original config
                device_placement_handled_by_loader = True
                self.logger.info(f"Model '{model_name}' ({model_cfg.quantization}) is bitsandbytes quantized. Device placement handled by loader.")
            elif hasattr(model, 'hf_device_map') and model.hf_device_map is not None:
                device_placement_handled_by_loader = True
                self.logger.info(f"Model '{model_name}' was loaded with a device map. Device placement handled by loader. Device map: {model.hf_device_map}")

            if not isinstance(model, torch.nn.DataParallel) and hasattr(model, 'to') and not device_placement_handled_by_loader:
                self.logger.info(f"Moving model '{model_name}' to device '{self.device}'.")
                model.to(self.device)
            
            # Determine final device for logging
            final_device_info = "N/A"
            if hasattr(model, 'device'):
                final_device_info = str(model.device)
            elif hasattr(model, 'hf_device_map') and model.hf_device_map:
                 # For models with device_map, report the map or a summary
                final_device_info = f"distributed via hf_device_map: {model.hf_device_map}"
            elif device_placement_handled_by_loader: # Fallback if specific attribute not found but placement was handled
                final_device_info = f"Handled by loader, target benchmark device: {self.device}"
            else: # If it was moved by .to(self.device)
                final_device_info = self.device

            self.logger.info(f"Model '{model_name}' loaded. Effective device(s): {final_device_info}.")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load model '{model_name}': {e}. Skipping this model.", exc_info=True)
            return None

    def _cleanup_model_resources(self, model: Optional[Any], tokenizer: Optional[Any]):
        """
        Releases model and tokenizer resources.
        :param model: The model instance to be cleaned up.
        :param tokenizer: The tokenizer instance to be cleaned up.
        """
        model_name = getattr(model, 'name_or_path', 'Unknown') # Try to get name for log
        self.logger.info(f"Cleaning up resources for model '{model_name}'...")
        del model
        del tokenizer
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.logger.info(f"Resources cleaned up for model '{model_name}'.")


    def _process_task_batches(self, handler: Any, data_loader: DataLoader, task_name: str, evaluator: Evaluator) -> bool:
        """
        Iterates through DataLoader, processes batches, and updates evaluator with results.
        Logs intermediate metric scores periodically.

        :param handler: Task handler instance for processing batches.
        :param data_loader: DataLoader for the dataset (can be a tqdm iterator).
        :param task_name: Name of the task being processed.
        :param evaluator: Evaluator instance for updating metrics.
        :return: True if processing was successful, False if errors occurred that stopped processing.
        """
        batch_num = 0
        log_interval_to_use = 0  # Default to 0 (no intermediate logging)

        # Retrieve log_interval from EvaluationConfig instance
        if self.config.evaluation: # self.config.evaluation should be an EvaluationConfig instance
            configured_log_interval = self.config.evaluation.log_interval

            if configured_log_interval is not None:
                if isinstance(configured_log_interval, int) and configured_log_interval > 0:
                    log_interval_to_use = configured_log_interval
                else:
                    # Log a warning if the configured value is invalid but present
                    self.logger.warning(
                        f"Invalid log_interval value '{configured_log_interval}' in evaluation config "
                        f"for task '{task_name}'. Disabling intermediate metric logging."
                    )

        self.logger.info(
            f"Starting batch processing for task '{task_name}'. "
            f"Intermediate log interval: {log_interval_to_use if log_interval_to_use > 0 else 'Disabled'}"
        )
        
        for batch in data_loader:  # data_loader is expected to be iterable (e.g., DataLoader or tqdm(DataLoader))
            batch_num += 1
            try:
                processed_outputs_from_handler = handler.process_batch(batch)
                # processed_outputs_from_handler should be a tuple (predictions, labels) or a dict
                if isinstance(processed_outputs_from_handler, dict) and "text_predictions" in processed_outputs_from_handler : # È l'output di GSM8K
                    # Assuming the handler returns a dict with keys "text_predictions" and "labels_for_text"
                    evaluator.update_batch_metrics(processed_outputs_from_handler, task_name=task_name) # Passa anche task_name
                elif isinstance(processed_outputs_from_handler, tuple) and len(processed_outputs_from_handler) == 2:
                    # Assuming the handler returns a tuple (predictions, labels)
                    predictions_batch, labels_batch = processed_outputs_from_handler
                    if predictions_batch is not None and not isinstance(predictions_batch, list):
                        predictions_batch = [predictions_batch] 
                    if labels_batch is not None and not isinstance(labels_batch, list):
                        labels_batch = [labels_batch]

                    if predictions_batch is not None and labels_batch is not None:
                        if len(predictions_batch) == len(labels_batch):
                            evaluator.update_batch_metrics((predictions_batch, labels_batch), task_name=task_name) # Passa come tupla
                        else:
                            self.logger.warning(
                                f"Batch {batch_num} for task '{task_name}': predictions ({len(predictions_batch)}) "
                                f"and labels ({len(labels_batch)}) length mismatch. Skipping batch for evaluation."
                            )
                    elif predictions_batch is None or labels_batch is None:
                         self.logger.warning(
                            f"Batch {batch_num} for task '{task_name}' produced None for predictions or labels. "
                            f"Skipping metric update for this batch."
                        )
                else:
                    self.logger.error(f"Task '{task_name}' handler returned unexpected output type: {type(processed_outputs_from_handler)}")

                # Log intermediate metrics
                if log_interval_to_use > 0 and batch_num % log_interval_to_use == 0:
                    intermediate_results = {}
                    # Accessing evaluator._metrics_instances directly for logging purposes.
                    # Consider adding a public method to Evaluator if this feels too coupled.
                    for metric_name, metric_instance in evaluator._metrics_instances.items():
                        try:
                            current_value = metric_instance.result()
                            # Simple formatting for the log
                            if isinstance(current_value, float):
                                formatted_value = f"{current_value:.4f}"
                            elif isinstance(current_value, dict):
                                formatted_value = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in current_value.items()])
                            else:
                                formatted_value = str(current_value)
                            intermediate_results[metric_name] = formatted_value
                        except Exception as e:
                            self.logger.debug(f"Could not compute intermediate result for metric '{metric_name}': {e}")
                            intermediate_results[metric_name] = "Error"
                    
                    if intermediate_results:
                        log_msg_parts = [f"{name}: {value}" for name, value in intermediate_results.items()]
                        self.logger.info(
                            f"Task '{task_name}' - Batch {batch_num}/{getattr(data_loader, '__len__', '?')} - " # Show total batches if available
                            f"Intermediate Metrics: {{{', '.join(log_msg_parts)}}}"
                        )
                        
                        # Optional: Update tqdm postfix if pbar is passed and is a tqdm instance
                        # if isinstance(data_loader, tqdm) and hasattr(data_loader, 'set_postfix'):
                        #     data_loader.set_postfix(intermediate_results, refresh=True)


            except Exception as e:
                self.logger.error(
                    f"Error processing batch {batch_num} for task '{task_name}': {e}. Stopping task.", 
                    exc_info=True # Include traceback for easier debugging
                )
                return False  # Indicate failure

        self.logger.info(f"Finished batch processing for task '{task_name}'. Processed {batch_num} batches.")
        return True


    def _run_task_evaluation(self, model: Any, tokenizer: Any, task_cfg: TaskConfig, dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Runs a single task and evaluates it using batched metric updates.
        :param model: The model to be evaluated.
        :param tokenizer: The tokenizer associated with the model.
        :param task_cfg: Configuration for the task to be run.
        :param dataset_info: Information about the dataset to be used.
        :return: Evaluation results or None if an error occurred.
        """
        task_name = task_cfg.name
        task_type = task_cfg.type
        dataset = dataset_info["dataset"]
        model_identifier = model.name_or_path if hasattr(model, 'name_or_path') else 'Unknown Model'
        self.logger.info(f"Running task '{task_name}' for model '{model_identifier}'...")

        # 1. Get Task Handler
        current_advanced_args = self.config.advanced.model_dump(exclude_none=True) if self.config.advanced else {} # Pydantic V2+
        # Or: current_advanced_args = self.config.advanced.dict(exclude_none=True) if self.config.advanced else {} # Pydantic V1
        
        # --- Pass task-specific and dataset-specific info to handler ---
        # Add task-specific handler_options from TaskConfig
        if task_cfg.handler_options:
            current_advanced_args.update(task_cfg.handler_options)
            self.logger.debug(f"Added handler_options to advanced_args for task '{task_name}': {task_cfg.handler_options}")

        if task_cfg.datasets:
            first_dataset_cfg = task_cfg.datasets[0]
            if first_dataset_cfg.config:
                current_advanced_args["dataset_config_name"] = first_dataset_cfg.config
            current_advanced_args["dataset_name"] = first_dataset_cfg.name # e.g., "opus100", "glue"

        try:
            handler_key_to_use = task_cfg.handler if task_cfg.handler else task_type
            handler = TaskHandlerFactory.get_handler(
                handler_key_to_use, 
                model, 
                tokenizer, 
                self.device, 
                current_advanced_args # Pass the combined/enriched args
            )
            self.logger.debug(f"TaskHandler obtained for '{task_name}' with advanced_args: {current_advanced_args.keys()}")
        except ValueError as e:
            self.logger.error(f"Could not get handler for task '{task_name}' (type: {task_type}): {e}")
            return {"error": f"Handler not found: {e}"}

        # 2. Setup DataLoader
        batch_size = current_advanced_args.get('batch_size', 32)

        try:
            is_map_style = isinstance(dataset, Dataset) and not isinstance(dataset, IterableDataset)
            data_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(self.device == 'cuda')
            )
        except Exception as e:
            self.logger.error(f"Failed to create DataLoader for task '{task_name}': {e}.", exc_info=True)
            return {"error": f"DataLoader creation failed: {e}"}

        # 3. Initialize Evaluator and Prepare Metrics
        eval_params_dict = {}
        if self.config.evaluation: # self.config.evaluation is an EvaluationConfig instance
            eval_params_dict = self.config.evaluation.model_dump(exclude_none=True) # Pydantic V2+
        evaluator = Evaluator(evaluation_params=eval_params_dict)
        
        formatted_metrics_config = [m.model_dump(exclude_none=True) for m in task_cfg.evaluation_metrics] # Pydantic V2+
        # Or: formatted_metrics_config = [m.dict(exclude_none=True) for m in task_cfg.evaluation_metrics] # Pydantic V1
        evaluator.prepare_metrics(formatted_metrics_config)
        self.logger.debug(f"Evaluator prepared for task '{task_name}'.")

        # 4. Process Batches with Progress Bar
        total_batches = None
        if dataset and is_map_style and hasattr(dataset, '__len__'):
            try:
                dataset_len = len(dataset)
                total_batches = (dataset_len + batch_size - 1) // batch_size
            except TypeError:
                self.logger.warning(f"Could not determine length for map-style dataset '{task_name}'. Progress bar may be inaccurate.")

        progress_desc = f"Task '{task_name}' ({model_identifier})"
        evaluation_successful = True
        with tqdm(data_loader, total=total_batches, desc=progress_desc, unit="batch", leave=False) as pbar:
            evaluation_successful = self._process_task_batches(handler, pbar, task_name, evaluator)

        if not evaluation_successful:
            self.logger.warning(f"Task '{task_name}' on model '{model_identifier}' processing was not fully successful due to batch errors.")

        # 5. Finalize and Get Results from Evaluator
        try:
            evaluation_results = evaluator.finalize_results()
            if evaluation_results:
                log_msg_parts = []
                for metric_name, value in evaluation_results.items():
                    if isinstance(value, float):
                        log_msg_parts.append(f"{metric_name}: {value:.4f}")
                    elif isinstance(value, dict): 
                        dict_parts = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in value.items()])
                        log_msg_parts.append(f"{metric_name}: {{{dict_parts}}}")
                    else:
                        log_msg_parts.append(f"{metric_name}: {value}")
                
                self.logger.info(
                    f"Final Evaluation Results for Task '{task_name}' on model '{model_identifier}': "
                    f"{{{', '.join(log_msg_parts)}}}"
                )
            
            self.logger.info(f"Task '{task_name}' on model '{model_identifier}' evaluation completed. Metrics requested: {[m.name for m in task_cfg.evaluation_metrics]}")
            
            if not evaluation_results and evaluation_successful:
                self.logger.warning(f"Task '{task_name}' on model '{model_identifier}' produced no evaluation results, though batch processing reported success.")
                return {"status": "No evaluation results generated"}
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Failed to finalize evaluation for task '{task_name}' on model '{model_identifier}': {e}", exc_info=True)
            return {"error": f"Evaluation finalization failed: {e}"}
        

    def _save_results(self):
        """Saves the final benchmark results."""
        if not self.results:
            self.logger.warning("No benchmark results were generated to save.")
            return
        try:
            self.logger.info(f"Saving results to {self.output_dir} in {self.reporting_format} format...")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_results(
                results=self.results,
                output_dir=str(self.output_dir), # Ensure path is string
                format=self.reporting_format
            )
            self.logger.info("Results saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}", exc_info=True)


    def run(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Executes the benchmark: loads datasets, runs models on tasks, evaluates, and saves results.
        :return: Dictionary of results structured as {model: {task: {metric: value}}}
        """
        # Load all datasets first
        task_datasets = self._load_all_datasets()
        if not task_datasets:
            return {} # Abort if no datasets could be loaded

        # Iterate through models
        if not self.config.models:
             self.logger.error("No models defined in the configuration. Aborting.")
             return {}

        for model_cfg in self.config.models:
            model_tokenizer_tuple = self._load_model_and_tokenizer(model_cfg)
            if model_tokenizer_tuple is None:
                continue # Skip model if loading failed

            model, tokenizer = model_tokenizer_tuple
            model_name = model_cfg.name
            self.results[model_name] = {} # Initialize results for this model

            # Iterate through tasks for the current model
            for task_cfg in self.config.tasks:
                task_name = task_cfg.name
                # Check if dataset for this task is available
                if task_name not in task_datasets or task_datasets[task_name] is None:
                    self.logger.warning(f"Skipping task '{task_name}' for model '{model_name}' (dataset not loaded).")
                    continue

                dataset_info = task_datasets[task_name]
                try:
                    # Run task and evaluate
                    evaluation_results = self._run_task_evaluation(model, tokenizer, task_cfg, dataset_info)

                    # Store results (or error)
                    if evaluation_results is not None:
                         self.results[model_name][task_name] = evaluation_results
                    else:
                         # Optionally store a marker indicating no results were produced
                         self.results[model_name][task_name] = {"status": "No results generated"}
                except Exception as e:
                    # Catch unexpected errors during the task run/evaluation call itself
                    self.logger.error(f"Unexpected error during task execution/evaluation for '{task_name}' on model '{model_name}': {e}", exc_info=True)
                    self.results[model_name][task_name] = {"error": f"Unexpected task error: {e}"}

            # Clean up model resources after processing all tasks for it
            self._cleanup_model_resources(model, tokenizer)

        # Save final results after all models are processed
        self._save_results()
        return self.results