import importlib
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Callable
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset, IterableDataset
import os
from .base_dataset_loader import BaseDatasetLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Define standard field names expected by handlers for each task type
# This mapping drives the normalization logic
TASK_TYPE_STANDARD_FIELDS = {
    "summarization": {"input": "input_text", "target": "target_text"},
    "multiple_choice_qa": {"question": "question", "choices": "choices", "label": "label_index", "subject": "subject"},
    "math_reasoning_generation": {"input": "input_text", "target": "target_text"},
    "translation": {"input": "input_text", "target": "target_text"},
    "classification": {"input": "input_text", "label": "label_index"}, # For single sentence classification like SST-2, CoLA
    "text_pair_classification": {"input1": "input_text_pair1", "input2": "input_text_pair2", "label": "label_index"} # For sentence-pair tasks
}

# Define potential original field names for common datasets
# This helps the normalizer guess the original names if hints aren't provided
DEFAULT_ORIGINAL_FIELD_MAP = {
    # General summarization (can be overridden by more specific ones)
    "summarization": {"input": "article", "target": "highlights"},
    # Specific for cnn_dailymail if its fields were different from general summarization
    "summarization_cnn_dailymail": {"input": "article", "target": "highlights"},

    "multiple_choice_qa": {"question": "question", "choices": "choices", "label": "answer", "subject": "subject"},

    "math_reasoning_generation_gsm8k": {"input": "question", "target": "answer"},

    # For GLUE tasks (using dataset config name for specificity)
    "classification_sst2": {"input": "sentence", "label": "label"},
    "classification_cola": {"input": "sentence", "label": "label"},
    "text_pair_classification_mrpc": {"input1": "sentence1", "input2": "sentence2", "label": "label"},
    "text_pair_classification_qqp": {"input1": "question1", "input2": "question2", "label": "label"},
    "text_pair_classification_mnli": {"input1": "premise", "input2": "hypothesis", "label": "label"},
    "text_pair_classification_qnli": {"input1": "question", "input2": "sentence", "label": "label"},
    "text_pair_classification_rte": {"input1": "sentence1", "input2": "sentence2", "label": "label"},
    # STS-B is regression, label is float, might need special handling for 'label_index' if that's always int
    "text_pair_classification_stsb": {"input1": "sentence1", "input2": "sentence2", "label": "label"},

    # NOTE: "translation" for opus100 is handled by a special path in _normalize_dataset
    # because it requires extraction from a nested dict, not just renaming.
}

class ConcreteDatasetLoader(BaseDatasetLoader):
    _SOURCE_REGISTRY: Dict[str, Callable] = {}

    def __init__(self, name: str, source_type: str = "hf_hub",
                 config: Optional[str] = None, split: str = "train",
                 data_dir: Optional[str] = None, streaming: bool = False,
                 dataset_specific_fields: Optional[Dict[str, str]] = None,
                 loader_args: Optional[Dict[str, Any]] = None,
                 max_samples: Optional[int] = None, **loader_kwargs):
        """
        Initializes the dataset loader with the specified parameters.
        :param name: Dataset name or type (e.g., "glue", "csv")
        :param source_type: Source type ("hf_hub", "local", or custom)
        :param config: Dataset subset/configuration
        :param split: Data split to load
        :param data_dir: Local directory or file path
        :param streaming: Enable streaming mode
        :param dataset_specific_fields: Explicit mapping for input/target fields
        :param max_samples: Maximum number of samples to load
        :param loader_kwargs: Additional kwargs for dataset loader
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid dataset name")

        self.name = name
        self.source_type = source_type
        self.config = config
        self.split = split
        self.data_dir = data_dir
        self.streaming = streaming
        self.dataset_specific_fields = dataset_specific_fields or {}
        self.loader_kwargs = loader_kwargs
        self.max_samples = max_samples

        if loader_args and isinstance(loader_args, dict):
            self.loader_kwargs.update(loader_args)
        
        logger.debug(f"ConcreteDatasetLoader for '{name}' initialized. "
                f"Effective loader_kwargs for hf_load_dataset: {self.loader_kwargs}")

    @classmethod
    def register_source(cls, source_type: str, loader_fn: Callable):
        """Registers a custom loader function."""
        cls._SOURCE_REGISTRY[source_type.lower()] = loader_fn

    def load(self, task_type: Optional[str] = None) -> Iterable:
        """
        Loads and optionally normalizes the dataset.
        :param task_type: Task type for field normalization
        :return: Loaded (and optionally normalized) dataset
        """
        logger.info(f"Loading dataset: {self.name} (config: {self.config}, split: {self.split})")
        try:
            dataset = self._load_dataset_by_source()
            if not dataset:
                raise RuntimeError(f"Dataset loading failed: {self.name} (config: {self.config})")

            # Apply cutoff if max_samples is specified
            if self.max_samples is not None and self.max_samples > 0:
                original_length_info = "unknown (iterable)"
                if hasattr(dataset, "__len__"):
                    original_length_info = str(len(dataset))

                if isinstance(dataset, IterableDataset):
                    logger.info(f"Applying cutoff: taking first {self.max_samples} samples from iterable dataset '{self.name}'. Original size: {original_length_info}.")
                    dataset = dataset.take(self.max_samples)
                elif hasattr(dataset, "select") and callable(dataset.select): # Map-style dataset
                    current_len = len(dataset)
                    if current_len > self.max_samples:
                        logger.info(f"Applying cutoff: selecting first {self.max_samples} samples from dataset '{self.name}'. Original size: {current_len}.")
                        dataset = dataset.select(range(self.max_samples))
                    else:
                        logger.info(f"Cutoff ({self.max_samples}) for dataset '{self.name}' is >= actual size ({current_len}). Using full dataset split.")
                else:
                    logger.warning(f"Dataset cutoff requested for '{self.name}', but dataset type does not support .select() or .take(). Cutoff not applied.")
            
            # Normalize after cutoff
            return self._normalize_dataset(dataset, task_type) if task_type else dataset
        except Exception as e:
            logger.error(f"Dataset loading/normalization/cutoff failed for '{self.name}': {e}", exc_info=True)
            raise

    def _load_dataset_by_source(self) -> Iterable:
        """
        Loads the dataset based on the specified source type.
        :return: Loaded dataset
        """
        source = self.source_type.lower()
        if source in self._SOURCE_REGISTRY:
            return self._SOURCE_REGISTRY[source](**self._base_loader_args())
        if source == "hf_hub":
            return self._load_hf_dataset()
        if source == "local":
            return self._load_local_files()
        raise ValueError(f"Unsupported source_type: {self.source_type}")

    def _base_loader_args(self) -> Dict:
        """
        Constructs base arguments for dataset loading.
        :return: Dictionary of base arguments
        """
        return {
            "name": self.name, "config": self.config,
            "split": self.split, "data_dir": self.data_dir,
            "streaming": self.streaming, **self.loader_kwargs
        }

    def _load_hf_dataset(self) -> Iterable:
        """
        Loads dataset from Hugging Face Hub.
        :return: Loaded dataset
        """
        return hf_load_dataset(
            path=self.name, name=self.config, split=self.split,
            streaming=self.streaming, **self.loader_kwargs
        )

    def _load_local_files(self) -> Iterable:
        """
        Loads dataset from local files or directories.
        :return: Loaded dataset
        """
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise ValueError(f"Invalid data dir: {self.data_dir}")
        is_dir = os.path.isdir(self.data_dir)
        return hf_load_dataset(
            path_or_type=self.name,
            data_dir=self.data_dir if is_dir else None,
            data_files=self.data_dir if not is_dir else None,
            split=self.split, streaming=self.streaming,
            **self.loader_kwargs
        )

    def _normalize_dataset(self, dataset: Iterable, task_type: str) -> Iterable:
        """
        Normalizes the dataset fields based on the task type.
        :param dataset: The dataset to normalize
        :param task_type: Type of ML task (e.g., classification, translation)
        :return: Normalized dataset
        """
        logger.debug(f"Normalizing dataset '{self.name}' (config: '{self.config}') for task '{task_type}'")
        if task_type not in TASK_TYPE_STANDARD_FIELDS:
            return dataset

        if task_type == "translation" and self.name == "opus100":
            return self._normalize_opus100(dataset)

        rename_map = self._build_rename_map(dataset, task_type)
        if not rename_map:
            return dataset

        return self._apply_renaming(dataset, rename_map, task_type)

    def _normalize_opus100(self, dataset: Iterable) -> Iterable:
        """
        Normalizes the OPUS-100 dataset by extracting input and target text from the translation field.
        :param dataset: The OPUS-100 dataset to normalize
        :return: Normalized dataset with input_text and target_text fields.
        """
        if not self.config or '-' not in self.config:
            logger.error(f"Missing or invalid OPUS-100 config: {self.config}")
            return dataset
        src, tgt = self.config.split('-', 1)

        def map_fn(example):
            trans = example.get("translation", {})
            return {
                "input_text": trans.get(src),
                "target_text": trans.get(tgt)
            } if src in trans and tgt in trans else {
                "input_text": None, "target_text": None
            }

        if isinstance(dataset, IterableDataset):
            logger.info("Applying map to IterableDataset for OPUS-100 (no caching for map function).")
            dataset = dataset.map(map_fn)
        else:
            dataset = dataset.map(map_fn, load_from_cache_file=True, remove_columns=list(dataset.column_names))

        dataset = dataset.filter(lambda ex: ex["input_text"] is not None and ex["input_text"] != "" and \
                                            ex["target_text"] is not None and ex["target_text"] != "")
        logger.info(f"OPUS-100 features after normalization: {dataset.features if hasattr(dataset, 'features') else 'IterableDataset (features not directly available)'}")
        return dataset

    def _build_rename_map(self, dataset: Iterable, task_type: str) -> Dict[str, str]:
        """
        Constructs a mapping of original field names to standard field names for the dataset.
        :param dataset: The dataset to inspect
        :param task_type: Type of ML task (e.g., classification, translation)
        :return: Mapping of original field names to standard field names
        """
        rename_map = {}
        standard_map = TASK_TYPE_STANDARD_FIELDS[task_type]
        columns = self._get_columns(dataset)

        for role, standard in standard_map.items():
            original = self._get_original_field_name(task_type, role)
            if original and original in columns and original != standard:
                rename_map[original] = standard
        return rename_map

    def _apply_renaming(self, dataset: Iterable, rename_map: Dict[str, str], task_type: str) -> Iterable:
        """
        Applies the renaming of fields in the dataset based on the provided mapping.
        :param dataset: The dataset to rename fields
        :param rename_map: Mapping of original field names to new field names
        :param task_type: Type of ML task (e.g., classification, translation)
        :return: Dataset with renamed fields
        """
        try:
            if isinstance(dataset, IterableDataset):
                logger.warning(f"Renaming fields via mapping for IterableDataset: {self.name} (no caching for map function)")

                def rename_iterable_example(ex): # Define the lambda as a proper function for clarity
                    new_ex = ex.copy() # Work on a copy
                    for old, new in rename_map.items():
                        if old in new_ex:
                            new_ex[new] = new_ex.pop(old)
                    return new_ex

                return dataset.map(rename_iterable_example)

            # For Map-style Dataset
            dataset = dataset.rename_columns(rename_map)
            logger.info(f"Applied rename_map: {rename_map}. Columns after rename: {list(dataset.column_names)}")

            # Determine columns to keep and remove for map-style datasets
            standard_fields_values = list(TASK_TYPE_STANDARD_FIELDS.get(task_type, {}).values())
            if not standard_fields_values: # Should not happen if task_type is valid
                logger.warning(f"No standard fields defined for task_type '{task_type}' when trying to remove columns.")
                return dataset

            columns_to_remove = [
                col for col in dataset.column_names if col not in standard_fields_values
            ]
            if columns_to_remove:
                logger.info(f"Removing columns after rename: {columns_to_remove}")
                dataset = dataset.remove_columns(columns_to_remove)
            else:
                logger.info(f"No columns to remove after renaming. Keeping: {list(dataset.column_names)}")

            return dataset
        except Exception as e:
            logger.error(f"Renaming or column removal error for task '{task_type}', dataset '{self.name}': {e}", exc_info=True)
            return dataset # Return original dataset on error

    def _get_columns(self, dataset: Iterable) -> List[str]:
        """
        Retrieves the column names from the dataset.
        :param dataset: The dataset to inspect
        :return: List of column names.
        """
        if hasattr(dataset, 'column_names'):
            return list(dataset.column_names)
        if hasattr(dataset, 'features'):
            return list(dataset.features.keys())
        return []

    def _get_original_field_name(self, task_type: str, standard_role: str) -> Optional[str]:
        """
        Determines the original field name in the dataset for a given standard role.
        Resolution priority:
        1. Explicit 'dataset_specific_fields' from YAML configuration.
        2. 'DEFAULT_ORIGINAL_FIELD_MAP' using a task-specific and dataset-identifier-specific key.
        3. 'DEFAULT_ORIGINAL_FIELD_MAP' using a general task-specific key.

        :param task_type: Type of ML task (e.g., "classification", "translation").
        :param standard_role: Logical field role to resolve (e.g., "input", "target", "label").
        :return: Original dataset field name if a mapping is found, otherwise None.
        """
        if standard_role in self.dataset_specific_fields:
            return self.dataset_specific_fields.get(standard_role)
        
        # 1. Check for explicit mapping in dataset_specific_fields
        dataset_identifier = self.config if self.name in ["glue", "super_glue"] and self.config else self.name
        
        # 2. Task-specific and dataset-identifier-specific mapping
        specific_map_key = f"{task_type}_{dataset_identifier}"
        if specific_map_key in DEFAULT_ORIGINAL_FIELD_MAP:
            task_specific_map = DEFAULT_ORIGINAL_FIELD_MAP[specific_map_key]
            if standard_role in task_specific_map:
                return task_specific_map[standard_role]

        # 3. Task-specific general mapping
        general_task_map_key = task_type
        if general_task_map_key in DEFAULT_ORIGINAL_FIELD_MAP:
            general_map = DEFAULT_ORIGINAL_FIELD_MAP[general_task_map_key]
            if standard_role in general_map:
                return general_map[standard_role]
        
        return None
    
class CustomScriptDatasetLoader(BaseDatasetLoader):
    """
    Loads a dataset by executing a user-defined function from a specified Python script.
    It expects 'script_path' and 'function_name' to be present in the kwargs passed
    to its constructor, which originate from the YAML configuration.
    """
    def __init__(self, 
                 name: str, 
                 source_type: str, # For consistency, though "custom_script" is implied
                 **loader_init_kwargs): # Catches ALL other args from YAML via factory
        """
        Initializes the CustomScriptDatasetLoader with the provided parameters.
        :param name: Name of the dataset
        :param source_type: Source type (should be "custom_script")
        :param loader_init_kwargs: Additional parameters for the dataset loader
        """
        # If BaseDatasetLoader has an __init__ that takes these, call it.
        # Otherwise, assign them directly.
        # super().__init__(name, source_type, **loader_init_kwargs) 
        
        self.name = name
        self.source_type = source_type
        self.logger = setup_logger(f"{self.__class__.__name__}.{self.name}") # More specific logger

        # Extract necessary parameters from loader_init_kwargs
        script_path_str = loader_init_kwargs.get("script_path")
        self.function_name = loader_init_kwargs.get("function_name")

        if not script_path_str:
            raise ValueError(f"'{self.__class__.__name__}' for dataset '{name}' requires 'script_path' in its configuration.")
        if not self.function_name:
            raise ValueError(f"'{self.__class__.__name__}' for dataset '{name}' requires 'function_name' in its configuration.")

        self.script_path = Path(script_path_str)
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Custom dataset script not found: {self.script_path}")

        # Store other relevant parameters from YAML, providing defaults
        self.config_param = loader_init_kwargs.get("config") # Dataset subset config
        self.split_param = loader_init_kwargs.get("split", "train")
        self.data_dir_param = loader_init_kwargs.get("data_dir")
        self.streaming = loader_init_kwargs.get("streaming", False)
        self.dataset_specific_fields = loader_init_kwargs.get("dataset_specific_fields", {})
        self.max_samples = loader_init_kwargs.get("max_samples")
        self.script_args = loader_init_kwargs.get("script_args", {})
        
        # Store any other unexpected kwargs if BaseDatasetLoader's super init doesn't.
        # This helps if BaseDatasetLoader uses **kwargs.
        self.other_kwargs = {k: v for k, v in loader_init_kwargs.items() if k not in [
            "script_path", "function_name", "config", "split", "data_dir", "streaming",
            "dataset_specific_fields", "max_samples", "script_args"
        ]}


        self.logger.info(
            f"Initialized CustomScriptDatasetLoader for '{self.name}': "
            f"script='{self.script_path}', function='{self.function_name}'"
        )
        self.logger.debug(f"  Split: {self.split_param}, Data Dir: {self.data_dir_param}, Max Samples: {self.max_samples}")
        self.logger.debug(f"  Script Args: {self.script_args}")
        self.logger.debug(f"  Dataset Specific Fields: {self.dataset_specific_fields}")
        self.logger.debug(f"  Other Kwargs received by init: {self.other_kwargs}")


    def load(self, task_type: Optional[str] = None) -> Iterable:
        """
        Loads the dataset by executing the user-defined function from the script.
        :param task_type: Type of ML task (e.g., classification, translation)
        :return: Loaded dataset
        """
        self.logger.info(f"Loading custom dataset '{self.name}' using script '{str(self.script_path)}' and function '{self.function_name}'")

        try:
            module_name = f"custom_dataset_module_{self.name.replace('-', '_')}_{self.function_name}"
            spec = importlib.util.spec_from_file_location(module_name, str(self.script_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {self.script_path}")

            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module) # type: ignore

            if not hasattr(custom_module, self.function_name):
                raise AttributeError(f"Function '{self.function_name}' not found in script '{str(self.script_path)}'")

            load_fn = getattr(custom_module, self.function_name)

            # Prepare arguments to pass to the user's script function
            args_for_user_function = {
                "split": self.split_param,
                "data_file_path": self.data_dir_param, 
                # Add any other parameters the user function explicitly expects by name
            }
            # Merge script_args from YAML (these are specific to the user's function)
            if self.script_args:
                args_for_user_function.update(self.script_args)
            
            # Pass only what the user function expects, or pass all if it takes **kwargs
            self.logger.debug(f"Calling custom dataset function '{self.function_name}' with args: {args_for_user_function}")
            dataset = load_fn(**args_for_user_function) # User function must accept these

            if dataset is None:
                raise ValueError(f"Custom dataset function '{self.function_name}' returned None.")

        except Exception as e:
            self.logger.error(f"Failed to load dataset from custom script '{str(self.script_path)}': {e}", exc_info=True)
            raise

        # Apply cutoff
        if self.max_samples is not None and self.max_samples > 0:
            # ... (cutoff logic as provided in your previous snippet, using self.name, self.max_samples) ...
            original_length_info = "unknown (iterable)"
            if hasattr(dataset, "__len__"):
                original_length_info = str(len(dataset))

            if isinstance(dataset, IterableDataset):
                self.logger.info(f"Applying cutoff: taking first {self.max_samples} samples from custom iterable dataset '{self.name}'. Original size: {original_length_info}.")
                dataset = dataset.take(self.max_samples)
            elif hasattr(dataset, "select") and callable(dataset.select):
                current_len = len(dataset)
                if current_len > self.max_samples:
                    self.logger.info(f"Applying cutoff: selecting first {self.max_samples} samples from custom map-style dataset '{self.name}'. Original size: {current_len}.")
                    dataset = dataset.select(range(self.max_samples))
            elif isinstance(dataset, list):
                if len(dataset) > self.max_samples:
                    self.logger.info(f"Applying cutoff: taking first {self.max_samples} samples from custom list-based dataset '{self.name}'. Original size: {len(dataset)}.")
                    dataset = dataset[:self.max_samples]

        # Normalize dataset
        # This part needs ConcreteDatasetLoader to be importable and its _normalize_dataset to be callable.
        # A cleaner way would be to have a standalone normalizer utility.
        if task_type:
            # Assuming ConcreteDatasetLoader's __init__ can handle being called with just these.
            # If not, this temporary instantiation needs to match ConcreteDatasetLoader's required args.
            temp_normalizer_loader = ConcreteDatasetLoader(
                name=self.name,
                source_type="temp_for_norm", # Placeholder
                dataset_specific_fields=self.dataset_specific_fields
                # Add other mandatory params for ConcreteDatasetLoader.__init__ if any, or make them optional.
            )
            normalized_dataset = temp_normalizer_loader._normalize_dataset(dataset, task_type)
        else:
            normalized_dataset = dataset
            
        self.logger.info(f"Custom dataset '{self.name}' loaded and potentially normalized for task_type '{task_type}'.")
        return normalized_dataset

    @classmethod
    def register_source(cls, source_type: str, loader_fn: Callable):
        """
        Registers a custom loader function for a specific source type.
        :param source_type: Source type identifier
        :param loader_fn: Function to load the dataset
        """
        logger.warning("CustomScriptDatasetLoader uses dynamic script loading; direct 'register_source' might not be typical.")