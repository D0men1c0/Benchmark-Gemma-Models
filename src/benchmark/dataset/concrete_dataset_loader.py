from typing import Dict, Any, Iterable, Optional, Callable
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset, IterableDataset
import os
from .base_dataset_loader import BaseDatasetLoader
from utils.logger import setup_logger # Assuming logger setup

logger = setup_logger(__name__)

# Define standard field names expected by handlers for each task type
# This mapping drives the normalization logic
TASK_TYPE_STANDARD_FIELDS = {
    "summarization": {"input": "input_text", "target": "target_text"},
    "multiple_choice_qa": {"question": "question", "choices": "choices", "label": "label_index"},
    "math_reasoning_generation": {"input": "input_text", "target": "target_text"},
    "translation": {"input": "input_text", "target": "target_text"}, # Needs lang hints potentially
    # Add other task types and their standard fields
}

# Define potential original field names for common datasets
# This helps the normalizer guess the original names if hints aren't provided
DEFAULT_ORIGINAL_FIELD_MAP = {
    "summarization": {"input": "article", "target": "highlights"},
    "multiple_choice_qa": {"question": "question", "choices": "choices", "label": "answer"}, # MMLU default
    "math_reasoning_generation": {"input": "question", "target": "answer"}, # GSM8K default
    "translation": {"data_key": "translation", "input_lang": "en", "target_lang": "fr"}, # Example for nested
}

class ConcreteDatasetLoader(BaseDatasetLoader):
    """Concrete implementation of dataset loader with normalization."""

    _SOURCE_REGISTRY: Dict[str, Callable] = {}

    def __init__(self, name: str, source_type: str = "hf_hub",
                config: Optional[str] = None, split: str = "train",
                data_dir: Optional[str] = None, streaming: bool = False,
                # Optional hints can be passed via loader_kwargs if needed from config
                **loader_kwargs):

        if not isinstance(name, str) or len(name.strip()) == 0:
            raise ValueError("Invalid dataset name")

        self.name = name
        self.source_type = source_type
        self.config = config
        self.split = split
        self.data_dir = data_dir
        self.streaming = streaming
        self.loader_kwargs = loader_kwargs
    
    @classmethod
    def register_source(cls, source_type: str, loader_fn: Callable):
        cls._SOURCE_REGISTRY[source_type.lower()] = loader_fn

    def load(self, task_type: Optional[str] = None) -> Iterable:
        """Load and normalize the dataset from the configured source."""
        source_type = self.source_type.lower()
        dataset: Optional[Iterable] = None

        try:
            if source_type in self._SOURCE_REGISTRY:
                # Custom loaders should ideally also perform normalization
                dataset = self._load_custom(source_type)
            elif source_type == "hf_hub":
                dataset = self._load_hf_dataset()
            elif source_type == "local":
                dataset = self._load_local_files()
            else:
                raise ValueError(f"Unsupported source: {source_type}")

            if dataset and task_type:
                # Apply normalization after loading
                dataset = self._normalize_dataset(dataset, task_type)
            elif not task_type:
                 logger.warning("task_type not provided to load(), skipping normalization.")

            if dataset is None:
                 raise RuntimeError("Dataset loading failed.")

            return dataset

        except Exception as e:
            logger.error(f"Failed during dataset loading or normalization for {self.name}: {e}", exc_info=True)
            raise # Re-raise the exception

    def _load_hf_dataset(self):
        return hf_load_dataset(
            self.name,
            name=self.config, # Use 'name' arg for HF config
            split=self.split,
            streaming=self.streaming,
            **self.loader_kwargs
        )

    def _load_local_files(self):
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise ValueError(f"Invalid data dir: {self.data_dir}")
        return hf_load_dataset(
            self.name, # Assumes format like 'csv', 'json' is the name
            data_dir=self.data_dir,
            split=self.split,
            streaming=self.streaming,
            **self.loader_kwargs
        )

    def _load_custom(self, source_type: str):
        return self._SOURCE_REGISTRY[source_type](
            name=self.name, config=self.config, split=self.split,
            data_dir=self.data_dir, streaming=self.streaming,
            **self.loader_kwargs
        )

    def _normalize_dataset(self, dataset: Iterable, task_type: str) -> Iterable:
        """Applies normalization (e.g., renaming fields) based on task type."""
        if task_type not in TASK_TYPE_STANDARD_FIELDS:
            logger.warning(f"No standard fields defined for task_type '{task_type}'. Returning dataset as is.")
            return dataset

        standard_fields = TASK_TYPE_STANDARD_FIELDS[task_type]
        default_originals = DEFAULT_ORIGINAL_FIELD_MAP.get(task_type, {})
        # hints = self.normalization_hints # Use hints if provided

        # This needs refinement for streaming datasets and complex types
        if isinstance(dataset, (Dataset, IterableDataset)):
            try:
                # Determine the columns to rename
                rename_map = {}
                current_columns = []
                if hasattr(dataset, 'column_names'):
                    current_columns = dataset.column_names
                elif hasattr(dataset, 'features') and dataset.features:
                    current_columns = list(dataset.features.keys())

                if not current_columns and isinstance(dataset, IterableDataset):
                    # For iterable datasets, we might need to inspect the first element
                    # This is less reliable and might consume the first element if not careful
                    logger.warning("Cannot reliably get column names for IterableDataset without inspection. Normalization might be incomplete.")
                    # Example: first_item = next(iter(dataset)) if dataset else None
                    #          if first_item: current_columns = list(first_item.keys()) ... then need to prepend first_item
                    # For now, we'll rely on defaults/hints for iterable if no features available
                elif not current_columns:
                    logger.warning(f"Could not determine columns for dataset {self.name}. Skipping renaming.")
                    return dataset

                # Map standard roles (input, target, etc.) to original names
                # Priority: Hint -> Default -> Standard Name itself
                for standard_role, standard_name in standard_fields.items():
                    # original_name = hints.get(standard_role, default_originals.get(standard_role, standard_name))
                    # Simplified: Use default original name if available, else assume standard name exists
                    original_name = default_originals.get(standard_role, standard_name)

                    if original_name in current_columns and original_name != standard_name:
                        rename_map[original_name] = standard_name
                    elif standard_name not in current_columns and original_name not in current_columns:
                        logger.warning(f"Field for standard role '{standard_role}' ('{standard_name}' or default '{original_name}') not found in dataset columns: {current_columns} for task type '{task_type}'.")

                if rename_map:
                    logger.info(f"Applying field renaming for task '{task_type}': {rename_map}")
                    if isinstance(dataset, Dataset): # Map-style dataset
                        # remove_columns needs care not to remove something needed by another rename
                        cols_to_remove = [orig for orig in rename_map.keys() if orig not in rename_map.values()]
                        return dataset.rename_columns(rename_map).remove_columns(cols_to_remove)
                    elif isinstance(dataset, IterableDataset) and hasattr(dataset, 'rename_columns'):
                        # rename_columns might not be available or fully functional on all iterables
                        cols_to_remove = [orig for orig in rename_map.keys() if orig not in rename_map.values()]
                        try:
                            # Note: remove_columns might also not work reliably on IterableDatasets
                            return dataset.rename_columns(rename_map) # .remove_columns(cols_to_remove)
                        except Exception as map_err:
                            logger.error(f"Failed to apply rename_columns on IterableDataset: {map_err}. Falling back to per-example mapping.")
                            # Fallback to map if rename fails (slower for iterable)
                            def map_example(example):
                                new_example = example.copy()
                                for old, new in rename_map.items():
                                    if old in new_example:
                                        new_example[new] = new_example.pop(old)
                                return new_example
                            return dataset.map(map_example) # remove_columns difficult here
                    else:
                        logger.warning("Dataset type does not support efficient renaming. Skipping.")

            except Exception as e:
                logger.error(f"Error during dataset field normalization for task '{task_type}': {e}", exc_info=True)
                # Decide whether to return original dataset or raise error
                return dataset # Return original on error

        return dataset