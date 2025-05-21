import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datasets import Dataset
from src.benchmark.dataset.dataset_factory import DatasetFactory #
from src.benchmark.dataset.concrete_dataset_loader import ConcreteDatasetLoader, CustomScriptDatasetLoader #
from src.benchmark.dataset.base_dataset_loader import BaseDatasetLoader


# Simple Tests
def test_dataset_factory_get_hf_hub_loader():
    """Test if the factory returns a ConcreteDatasetLoader for hf_hub."""
    config = {"name": "test_ds", "source_type": "hf_hub"}
    loader = DatasetFactory.from_config(config)
    assert isinstance(loader, ConcreteDatasetLoader)
    assert loader.name == "test_ds"
    assert loader.source_type == "hf_hub"

def test_dataset_factory_get_local_loader():
    """Test if the factory returns a ConcreteDatasetLoader for local."""
    config = {"name": "test_ds_local", "source_type": "local", "data_dir": "/tmp/data"}
    loader = DatasetFactory.from_config(config)
    assert isinstance(loader, ConcreteDatasetLoader)
    assert loader.data_dir == "/tmp/data"

def test_dataset_factory_get_custom_script_loader(tmp_path):
    """Test if the factory returns a CustomScriptDatasetLoader."""
    dummy_script_file = tmp_path / "dummy_script.py"
    dummy_script_file.touch()

    config = {
        "name": "custom_ds",
        "source_type": "custom_script",
        "script_path": str(dummy_script_file),
        "function_name": "load_fn"
    }
    loader = DatasetFactory.from_config(config)
    assert isinstance(loader, CustomScriptDatasetLoader)
    assert loader.script_path == dummy_script_file
    assert loader.function_name == "load_fn"

def test_dataset_factory_unsupported_type():
    """Test factory raises error for unsupported source_type."""
    config = {"name": "test_ds", "source_type": "unsupported_type"}
    with pytest.raises(ValueError, match="Unsupported dataset source_type"):
        DatasetFactory.from_config(config)

# More Complex Tests for ConcreteDatasetLoader (using mocks)
@patch('src.benchmark.dataset.concrete_dataset_loader.hf_load_dataset')
def test_concrete_loader_hf_hub_load_calls_hf_load_dataset(mock_hf_load):
    """Test ConcreteDatasetLoader calls hf_load_dataset for hf_hub source."""
    mock_dataset_iterable = MagicMock()
    mock_hf_load.return_value = mock_dataset_iterable

    loader_config = {
        "name": "glue",
        "source_type": "hf_hub",
        "config": "mrpc",
        "split": "train",
        "streaming": False,
        "loader_args": {"trust_remote_code": True}
    }
    loader = ConcreteDatasetLoader(**loader_config)
    
    # Mock _normalize_dataset to prevent it from actually running,
    # as we are unit testing the load part here.
    with patch.object(loader, '_normalize_dataset', side_effect=lambda d, t: d) as mock_normalize:
        dataset = loader.load(task_type="classification")

    mock_hf_load.assert_called_once_with(
        path="glue",
        name="mrpc",
        split="train",
        streaming=False,
        trust_remote_code=True # from loader_args
    )
    assert dataset is mock_dataset_iterable
    mock_normalize.assert_called_once()


@patch('src.benchmark.dataset.concrete_dataset_loader.hf_load_dataset')
def test_concrete_loader_local_files_load(mock_hf_load, tmp_path):
    """Test ConcreteDatasetLoader for local file source."""
    mock_dataset_iterable = MagicMock()
    mock_hf_load.return_value = mock_dataset_iterable

    # Create a dummy data file
    dummy_data_file = tmp_path / "my_data.csv"
    dummy_data_file.touch()

    loader_config = {
        "name": "csv", # Loader type for local files
        "source_type": "local",
        "data_dir": str(dummy_data_file), # Path to the file
        "split": "train", # Split might be ignored by hf_load_dataset if data_files is a single file
    }
    loader = ConcreteDatasetLoader(**loader_config)
    with patch.object(loader, '_normalize_dataset', side_effect=lambda d, t: d):
        dataset = loader.load(task_type="classification")
    
    mock_hf_load.assert_called_once_with(
        path_or_type="csv",
        data_dir=None, # Because data_dir was a file, not a directory
        data_files=str(dummy_data_file),
        split="train",
        streaming=False # Default from init
    )
    assert dataset is mock_dataset_iterable


# Tests for CustomScriptDatasetLoader
@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_custom_script_loader_success(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    """Test successful loading and execution of a custom dataset script."""
    script_content = """
import pandas as pd
from datasets import Dataset
def my_custom_loader(split, data_file_path, script_args=None):
    # In a real scenario, this would load data from data_file_path
    if split == 'train':
        data = {'text': ['sample 1', 'sample 2'], 'label': [0, 1]}
    else:
        data = {'text': ['test sample 1'], 'label': [0]}
    return Dataset.from_pandas(pd.DataFrame(data))
"""
    custom_script_file = tmp_path / "my_ds_script.py"
    custom_script_file.write_text(script_content)

    # Setup mocks for importlib
    mock_spec = MagicMock()
    mock_spec.loader.exec_module = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec
    
    mock_custom_module = MagicMock()

    # This is how you'd make the function available on the mocked module
    mock_dataset_instance = MagicMock(spec=Dataset) 

    # Simulate the return value of the custom loader function
    mock_dataset_instance.column_names = ['text', 'label'] 
    mock_custom_module.my_custom_loader = MagicMock(return_value=mock_dataset_instance)
    mock_module_from_spec.return_value = mock_custom_module

    loader_config = {
        "name": "my_custom_data",
        "source_type": "custom_script",
        "script_path": str(custom_script_file),
        "function_name": "my_custom_loader",
        "split": "train",
        "data_dir": str(tmp_path / "dummy_data_dir"), 
        "script_args": {"param1": "test"}
    }
    loader = CustomScriptDatasetLoader(**loader_config)

    # Mock the _normalize_dataset method
    with patch.object(ConcreteDatasetLoader, '_normalize_dataset', side_effect=lambda d, t: d) as mock_normalize:
        dataset_result = loader.load(task_type="classification")

    mock_spec_from_file_location.assert_called_once()
    mock_module_from_spec.assert_called_once_with(mock_spec)

    mock_normalize.assert_called_once_with(mock_dataset_instance, "classification")
    assert dataset_result is mock_dataset_instance # Should be the MagicMock from the side_effect

    mock_custom_module.my_custom_loader.assert_called_once_with(
        split="train",
        data_file_path=str(tmp_path / "dummy_data_dir"),
        param1="test" 
    )


def test_custom_script_loader_file_not_found():
    """Test CustomScriptDatasetLoader when script_path does not exist."""
    loader_config = {
        "name": "my_custom_data",
        "source_type": "custom_script",
        "script_path": "non_existent_script.py",
        "function_name": "my_custom_loader"
    }
    with pytest.raises(FileNotFoundError):
        CustomScriptDatasetLoader(**loader_config)

def test_custom_script_loader_function_not_found(tmp_path):
    """Test CustomScriptDatasetLoader when function_name is not in the script."""
    custom_script_file = tmp_path / "another_script.py"
    custom_script_file.write_text("def another_function(): pass") # Script exists, function doesn't

    loader_config = {
        "name": "my_custom_data",
        "source_type": "custom_script",
        "script_path": str(custom_script_file),
        "function_name": "missing_function"
    }
    loader = CustomScriptDatasetLoader(**loader_config) # Initialization should be fine
    with pytest.raises(AttributeError, match="Function 'missing_function' not found"):
        loader.load()