# data_utils.py
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import io # Required for type hinting and handling BytesIO from Streamlit

# to_float function remains the same
def to_float(value: Any) -> Optional[float]:
    """
    Convert a value to float, returning None if conversion fails.

    :param value: The value to convert.
    :type value: Any
    :return: The converted float value, or None if conversion is not possible.
    :rtype: Optional[float]
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

# flatten_json_results function remains the same
def flatten_json_results(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten the nested JSON structure from benchmark results into a Pandas DataFrame.
    Each metric, including sub-metrics of composite scores, becomes a separate row.

    :param data: Nested dictionary containing benchmark results.
                 Expected structure: {model_name: {task_name: {metric_name: score_value or {sub_metric: score_value}}}}
    :type data: Dict[str, Dict[str, Any]]
    :return: A DataFrame with columns: 'model', 'task', 'metric', 'sub_metric', 
             'score', and 'full_metric_name'.
    :rtype: pd.DataFrame
    """
    records: List[Dict[str, Any]] = []
    for model_name, tasks_data in data.items():
        if not isinstance(tasks_data, dict):
            print(f"Warning: Data for model '{model_name}' is not a dictionary, skipping.")
            continue
        for task_name, metrics_data in tasks_data.items():
            if not isinstance(metrics_data, dict):
                print(f"Warning: Data for task '{task_name}' in model '{model_name}' is not a dictionary, skipping.")
                continue
            
            base_info: Dict[str, Any] = {"model": model_name, "task": task_name}
            
            if "error" in metrics_data:
                print(f"Error recorded in JSON for {model_name}/{task_name}: {metrics_data['error']}")
                records.append({
                    **base_info, "metric": "processing_error",
                    "sub_metric": metrics_data['error'][:100], "score": None 
                })
                continue

            for metric_name, value in metrics_data.items():
                if metric_name == "error": continue
                if isinstance(value, dict):
                    for sub_metric_name, score_value in value.items():
                        records.append({
                            **base_info, "metric": metric_name,
                            "sub_metric": sub_metric_name, "score": to_float(score_value)
                        })
                else:
                    records.append({
                        **base_info, "metric": metric_name,
                        "sub_metric": None, "score": to_float(value)
                    })
    
    if not records:
        return pd.DataFrame(columns=["model", "task", "metric", "sub_metric", "score", "full_metric_name"])

    df = pd.DataFrame(records)
    if 'metric' in df.columns:
        df["full_metric_name"] = df.apply(
            lambda r: f"{r['metric']}_{r['sub_metric']}" if pd.notna(r['sub_metric']) and r['sub_metric'] else str(r['metric']),
            axis=1
        )
    else:
        df["full_metric_name"] = None 
        print("Warning: 'metric' column not found. 'full_metric_name' will be None.")
    return df

# MODIFIED FUNCTION:
def load_and_process_benchmark_data(file_input: Union[str, Path, io.BytesIO, io.TextIOWrapper]) -> Optional[pd.DataFrame]:
    """
    Load benchmark results from a JSON file (path) or an in-memory buffer (uploaded file)
    and process them into a DataFrame.

    :param file_input: Path to the JSON file (str or Path) or an in-memory file-like object
                       (e.g., io.BytesIO or io.TextIOWrapper from st.file_uploader).
    :type file_input: Union[str, Path, io.BytesIO, io.TextIOWrapper]
    :return: A Pandas DataFrame with the processed benchmark data, or None if loading fails.
    :rtype: Optional[pd.DataFrame]
    """
    raw_data = None
    source_description = ""

    try:
        if isinstance(file_input, (str, Path)):
            # Handling a file path
            path = Path(file_input)
            source_description = f"from file path: {path}"
            if not path.exists():
                print(f"Error: Benchmark JSON file not found at {path}")
                return None
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        # Handling an uploaded file object (which can be BytesIO or TextIOWrapper)
        elif isinstance(file_input, io.BytesIO): # Typically from st.file_uploader
            source_description = "from uploaded file (BytesIO)"
            file_input.seek(0) # Ensure reading from the beginning
            # st.file_uploader returns BytesIO, so decode bytes to string
            raw_data = json.load(io.TextIOWrapper(file_input, encoding='utf-8'))
        elif isinstance(file_input, io.TextIOWrapper): # If already a text wrapper
            source_description = "from uploaded file (TextIOWrapper)"
            file_input.seek(0)
            raw_data = json.load(file_input)
        elif hasattr(file_input, "read"): # More generic check for file-like objects
            source_description = f"from uploaded file-like object (type: {type(file_input).__name__})"
            file_input.seek(0)
            # Try decoding if it's bytes, otherwise assume text
            try:
                content = file_input.read()
                if isinstance(content, bytes):
                    raw_data = json.loads(content.decode('utf-8'))
                elif isinstance(content, str):
                    raw_data = json.loads(content)
                else: # Should not happen with standard file objects
                    print(f"Error: Uploaded file content has unexpected type: {type(content)}")
                    return None
            except Exception as e_read:
                print(f"Error reading/decoding uploaded file content: {e_read}")
                return None
        else:
            print(f"Error: Unsupported file_input type: {type(file_input)}")
            return None

        if raw_data is None:
            print(f"Error: Failed to load raw_data {source_description}.")
            return None

        df = flatten_json_results(raw_data)
        if df.empty and raw_data:
            print(f"Warning: Processed DataFrame is empty, though JSON {source_description} was read. Check structure.")
            return pd.DataFrame()
        elif not raw_data:
            print(f"Warning: JSON data {source_description} is empty.")
            return pd.DataFrame()
        return df
        
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON {source_description}. Error: {e}. Ensure valid JSON format.")
        return None
    except Exception as e:
        print(f"Error: Unexpected error loading/processing data {source_description}: {e}")
        return None


if __name__ == '__main__':
    # Example usage (for testing data_utils.py directly)
    CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT_FOR_TESTING = CURRENT_SCRIPT_DIR.parent 
    dummy_dir_project_level = PROJECT_ROOT_FOR_TESTING / "benchmarks_output"
    dummy_dir_project_level.mkdir(exist_ok=True)
    dummy_file_path_project_level = dummy_dir_project_level / "benchmark_results.json"

    if not dummy_file_path_project_level.exists():
        print(f"Creating dummy data for testing at: {dummy_file_path_project_level}")
        dummy_data = {
            "test_model_A": {"task_1": {"accuracy": 0.7, "f1": 0.65}},
            "test_model_B": {"task_1": {"accuracy": 0.75, "f1": 0.70}}
        }
        with open(dummy_file_path_project_level, 'w') as f: json.dump(dummy_data, f, indent=4)
    
    print(f"--- Testing data_utils.py directly ---")
    print(f"Attempting to load data from project root: {dummy_file_path_project_level}")
    df_results = load_and_process_benchmark_data(dummy_file_path_project_level)

    if df_results is not None:
        print("\nProcessed DataFrame Head:")
        print(df_results.head())
    else:
        print("\nFailed to load or process data during direct test.")