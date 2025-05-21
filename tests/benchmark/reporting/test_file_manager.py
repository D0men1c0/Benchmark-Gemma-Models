import pytest
import json
import yaml
import pickle
import csv
import pandas as pd
from fpdf import FPDF # For checking PDF content (basic check)
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from src.benchmark.reporting.file_manager import save_results, _flatten_results # Import internal for specific test
from src.utils.logger import setup_logger # Assuming logger is used and we might want to mock it

# Sample results data for testing
@pytest.fixture
def sample_results_data():
    return {
        "model_A": {
            "task_1": {"metric_X": 0.95, "metric_Y": 0.88},
            "task_2": {"metric_X": 0.92, "error": "failed to load"}
        },
        "model_B": {
            "task_1": {"metric_X": 0.90, "metric_Y": 0.85, "metric_Z": {"sub_1": 0.5, "sub_2": 0.6}}
        },
        "model_C": { # Model with non-dict task results (e.g., overall error)
            "error": "Model C failed entirely"
        }
    }

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "test_results"
    output_dir.mkdir()
    return output_dir

# Test the main save_results function for each format
@pytest.mark.parametrize("file_format, expected_extension, load_func", [
    ("json", "json", json.load),
    ("yaml", "yaml", yaml.safe_load),
    ("pickle", "pkl", pickle.load),
    # CSV and Parquet need more specific checks due to flattening
])
def test_save_results_formats_simple_load(sample_results_data, temp_output_dir, file_format, expected_extension, load_func):
    """Test saving in various formats that can be directly loaded and compared."""
    save_results(results=sample_results_data, output_dir=str(temp_output_dir), format=file_format)
    
    expected_file = temp_output_dir / f"benchmark_results.{expected_extension}"
    assert expected_file.exists()

    # For pickle, open in binary mode
    read_mode = "rb" if file_format == "pickle" else "r"
    with open(expected_file, read_mode) as f:
        loaded_data = load_func(f)
    
    assert loaded_data == sample_results_data

def test_save_results_csv(sample_results_data, temp_output_dir):
    """Test saving in CSV format with content verification."""
    save_results(results=sample_results_data, output_dir=str(temp_output_dir), format="csv")
    expected_file = temp_output_dir / "benchmark_results.csv"
    assert expected_file.exists()

    with open(expected_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Expected flattened structure:
    # model_A, task_1, metric_X=0.95, metric_Y=0.88, error=None, metric_Z=None
    # model_A, task_2, metric_X=0.92, metric_Y=None, error="failed to load", metric_Z=None
    # model_B, task_1, metric_X=0.90, metric_Y=0.85, error=None, metric_Z="{'sub_1': 0.5, 'sub_2': 0.6}"
    # model_C, task=None, metric_X=None, ... result="Model C failed entirely"

    assert len(rows) == 4 # 2 for model_A, 1 for model_B, 1 for model_C
    
    # Check a few key aspects, exact structure depends on _flatten_results
    # and how DictWriter handles nested dicts (it usually just str(dict))
    found_model_a_task_1 = any(
        r["model"] == "model_A" and r["task"] == "task_1" and float(r["metric_X"]) == 0.95 for r in rows
    )
    assert found_model_a_task_1

    found_model_b_task_1_z = any(
        r["model"] == "model_B" and r["task"] == "task_1" and "metric_Z" in r and "{'sub_1': 0.5, 'sub_2': 0.6}" in r["metric_Z"]
        for r in rows
    )
    assert found_model_b_task_1_z
    
    found_model_c_error = any(
        r["model"] == "model_C" and r.get("result") == "Model C failed entirely" for r in rows
    )
    assert found_model_c_error


def test_save_results_parquet(sample_results_data, temp_output_dir):
    """Test saving in Parquet format."""
    save_results(results=sample_results_data, output_dir=str(temp_output_dir), format="parquet")
    expected_file = temp_output_dir / "benchmark_results.parquet"
    assert expected_file.exists()

    df = pd.read_parquet(expected_file)
    assert len(df) == 4 # Matches the number of rows from _flatten_results
    # Verify some data points, similar to CSV
    model_a_task_1_df = df[(df["model"] == "model_A") & (df["task"] == "task_1")]
    assert model_a_task_1_df["metric_X"].iloc[0] == 0.95
    
    model_b_task_1_df = df[(df["model"] == "model_B") & (df["task"] == "task_1")]
    # Pandas will read the string representation of the dict for metric_Z if not further processed
    assert isinstance(model_b_task_1_df["metric_Z"].iloc[0], dict)
    assert model_b_task_1_df["metric_Z"].iloc[0] == {'sub_1': 0.5, 'sub_2': 0.6}

def test_save_results_pdf(sample_results_data, temp_output_dir):
    """Test saving in PDF format (basic existence and content check)."""
    save_results(results=sample_results_data, output_dir=str(temp_output_dir), format="pdf")
    expected_file = temp_output_dir / "benchmark_results.pdf"
    assert expected_file.exists()
    assert expected_file.stat().st_size > 100 # PDF should not be empty

    # Basic check: try to read some text from PDF
    # This is a very superficial check for PDF content.
    # More robust PDF content checking is complex.
    try:
        pdf = FPDF() # Need to instantiate to use its methods if we were reading.
                     # However, fpdf is for writing. Reading PDFs usually needs another lib.
                     # For now, just check if it was created and has some size.
        # To actually parse PDF content, you'd use libraries like PyPDF2 or pdfplumber
        # e.g. from PyPDF2 import PdfReader
        # reader = PdfReader(expected_file)
        # text_content = ""
        # for page in reader.pages:
        #     text_content += page.extract_text()
        # assert "Model: model_A" in text_content
        # assert "metric_X: 0.95" in text_content
        pass # Placeholder for more advanced PDF content check if needed
    except Exception as e:
        pytest.fail(f"Failed to perform basic check on PDF: {e}")


def test_save_results_unsupported_format(sample_results_data, temp_output_dir):
    """Test attempting to save in an unsupported format."""
    with pytest.raises(ValueError, match="Unsupported format: xyz. Choose from:"):
        save_results(results=sample_results_data, output_dir=str(temp_output_dir), format="xyz")

@patch("src.benchmark.reporting.file_manager.Path.mkdir")
def test_save_results_directory_creation(mock_mkdir, sample_results_data, tmp_path):
    """Test that the output directory is created."""
    output_dir_str = str(tmp_path / "new_results_dir")
    # Mock open to prevent actual file writing for this specific test
    with patch("builtins.open", mock_open()):
        save_results(results=sample_results_data, output_dir=output_dir_str, format="json")
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_flatten_results_logic():
    """Test the internal _flatten_results helper function."""
    data = {
        "model1": {"taskA": {"acc": 1.0, "loss": 0.1}, "taskB": {"acc": 0.9}},
        "model2": {"taskA": {"acc": 0.8, "loss": 0.2, "extra_metric": "val"}},
        "model3": {"error": "load_failed"}
    }
    flattened = _flatten_results(data)
    assert len(flattened) == 4 # 2 for model1, 1 for model2, 1 for model3
    
    expected_keys_m1_tA = {"model", "task", "acc", "loss"}
    expected_keys_m2_tA = {"model", "task", "acc", "loss", "extra_metric"}
    expected_keys_m3 = {"model", "result"}

    assert all(k in flattened[0] for k in expected_keys_m1_tA)
    assert flattened[0]["model"] == "model1" and flattened[0]["task"] == "taskA" and flattened[0]["acc"] == 1.0
    
    assert all(k in flattened[2] for k in expected_keys_m2_tA) # model2 results
    assert flattened[2]["model"] == "model2" and flattened[2]["task"] == "taskA" and flattened[2]["extra_metric"] == "val"

    assert all(k in flattened[3] for k in expected_keys_m3) # model3 results
    assert flattened[3]["model"] == "model3" and flattened[3]["result"] == "load_failed"

def test_save_csv_empty_results(temp_output_dir):
    """Test saving CSV when results are empty or lead to no flattened data."""
    with patch("src.benchmark.reporting.file_manager.logger.warning") as mock_logger_warning:
        save_results(results={}, output_dir=str(temp_output_dir), format="csv")
        expected_file = temp_output_dir / "benchmark_results.csv"
        # The current _save_csv implementation writes a header even if no data.
        # Depending on desired behavior, this could be changed.
        # For now, we just check the warning and that the file might exist (empty or header only).
        mock_logger_warning.assert_any_call("No data to save to CSV.")
        # File may or may not be created depending on strictness. If it is, it'd be header only.
        # assert not expected_file.exists() or expected_file.stat().st_size < 50 # Example check

        save_results(results={"model_error": "error"}, output_dir=str(temp_output_dir), format="csv")
        # This will create one row if _flatten_results handles it.
        assert expected_file.exists() # Should create file with "model", "result" header


def test_save_parquet_empty_results(temp_output_dir):
    """Test saving Parquet when results are empty."""
    with patch("src.benchmark.reporting.file_manager.logger.warning") as mock_logger_warning:
        save_results(results={}, output_dir=str(temp_output_dir), format="parquet")
        mock_logger_warning.assert_any_call("No data to save to Parquet.")
        expected_file = temp_output_dir / "benchmark_results.parquet"
        assert not expected_file.exists() # Pandas to_parquet raises error on empty DataFrame without columns.
                                        # If _flatten_results returns empty list, df would be empty.