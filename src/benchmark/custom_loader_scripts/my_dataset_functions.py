from typing import Any, Dict, Optional, Union
import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_simple_csv_data(split: str, 
                         data_file_path: str, # Path to the specific CSV file
                         script_args: Optional[Dict[str, Any]] = None) -> Union[Dataset, DatasetDict, None]:
    """
    Loads data from a single specified CSV file into a Hugging Face Dataset.
    'data_file_path' is expected to be passed via 'script_args' in the YAML.
    'split' might be part of the filename or ignored if the CSV is for one split.
    This function is called by CustomScriptDatasetLoader.
    :param split: The split name (e.g., 'train', 'validation', 'test').
    :param data_file_path: Path to the CSV file.
    :param script_args: Additional arguments passed from the YAML configuration.
    :return: A Hugging Face Dataset or DatasetDict.
    """
    if script_args is None:
        script_args = {}

    logger.info(f"load_simple_csv_data called for split: '{split}', file: '{data_file_path}'")
    logger.info(f"Additional script_args: {script_args}")

    file_path = Path(data_file_path)

    if not file_path.exists():
        logger.error(f"Error: CSV file not found at {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)

        hf_dataset = Dataset.from_pandas(df)
        logger.info(f"Loaded data from {file_path}. Features: {hf_dataset.features}")
        return hf_dataset
    except Exception as e:
        logger.error(f"Error loading or processing CSV {file_path}: {e}")
        return None

def load_multi_split_csv_from_dir(split: str, 
                                  data_path: str, # Path to the directory containing split CSVs
                                  script_args: Optional[Dict[str, Any]] = None) -> Union[Dataset, DatasetDict, None]:
    """
    Loads data from CSV files named like 'train.csv', 'validation.csv' in a directory.
    'data_path' is the directory, passed from 'data_dir' in YAML.
    'split' determines which CSV to load or if 'all' load a DatasetDict.
    This function is called by CustomScriptDatasetLoader.
    :param split: The split name (e.g., 'train', 'validation', 'test', or 'all').
    :param data_path: Directory containing the CSV files.
    :param script_args: Additional arguments passed from the YAML configuration.
    :return: A Hugging Face Dataset or DatasetDict.
    """
    if script_args is None:
        script_args = {}

    logger.info(f"load_multi_split_csv_from_dir called for split: '{split}', dir: '{data_path}'")
    base_data_dir = Path(data_path)

    if split.lower() == "all":
        all_datasets = {}
        for s_name in ["train", "validation", "test"]: # Example splits
            file_path = base_data_dir / f"{s_name}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                all_datasets[s_name] = Dataset.from_pandas(df)
        if not all_datasets: return None
        return DatasetDict(all_datasets)
    else:
        file_path = base_data_dir / f"{split}.csv"
        if not file_path.exists():
            logger.error(f"Error: CSV for split '{split}' not found at {file_path}")
            return None
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)