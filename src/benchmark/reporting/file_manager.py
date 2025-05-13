import json
import yaml
import pickle
import csv
from pathlib import Path
import pandas as pd
from fpdf import FPDF
from typing import Dict, Any, Callable, List

from utils.logger import setup_logger
logger = setup_logger()

# --- Saver Functions ---
def _save_json(results: Dict[str, Any], file_path: Path):
    """
    Save results to a JSON file.
    :param results: Dictionary containing benchmark results.
    :param file_path: Path to the output JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

def _save_yaml(results: Dict[str, Any], file_path: Path):
    """
    Save results to a YAML file.
    :param results: Dictionary containing benchmark results.
    :param file_path: Path to the output YAML file.
    """
    with open(file_path, "w") as f:
        yaml.safe_dump(results, f, default_flow_style=False)

def _flatten_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Helper to flatten results for tabular formats.
    :param results: Nested dictionary of benchmark results.
    :return: List of dictionaries, each representing a row in the table.
    """
    flattened = []
    for model, tasks in results.items():
        if isinstance(tasks, dict):
            for task, metrics in tasks.items():
                row = {"model": model, "task": task}
                if isinstance(metrics, dict):
                    row.update(metrics)
                else:
                    row["result"] = str(metrics)
                flattened.append(row)
        else:
            flattened.append({"model": model, "result": str(tasks)})
    return flattened


def _save_csv(results: Dict[str, Any], file_path: Path):
    """
    Save results to a CSV file.
    :param results: Dictionary containing benchmark results.
    :param file_path: Path to the output CSV file.
    """
    flattened = _flatten_results(results)
    if not flattened:
        logger.warning("No data to save to CSV.")
        return
    fieldnames = list(flattened[0].keys())
    all_keys = set().union(*(d.keys() for d in flattened))
    for row in flattened:
        for key in all_keys:
            row.setdefault(key, None)

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_keys))
        writer.writeheader()
        writer.writerows(flattened)


def _save_pickle(results: Dict[str, Any], file_path: Path):
    """
    Save results to a Pickle file.
    :param results: Dictionary containing benchmark results.
    :param file_path: Path to the output Pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(results, f)

def _save_parquet(results: Dict[str, Any], file_path: Path):
    """
    Save results to a Parquet file.
    :param results: Dictionary containing benchmark results.
    :param file_path: Path to the output Parquet file.
    """
    flattened = _flatten_results(results)
    if not flattened:
        logger.warning("No data to save to Parquet.")
        return
    df = pd.DataFrame(flattened)
    df.to_parquet(file_path, engine="pyarrow", index=False)


def _save_pdf(results: Dict[str, Any], file_path: Path):
    """
    Save results to a PDF file.
    :param results: Dictionary containing benchmark results.
    :param file_path: Path to the output PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Benchmark Results", ln=True, align="C")
    pdf.ln(10)

    for model, tasks in results.items():
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(200, 10, txt=f"Model: {model}", ln=True)
        pdf.set_font("Arial", size=10)
        if isinstance(tasks, dict):
            for task, metrics in tasks.items():
                pdf.cell(200, 8, txt=f"  Task: {task}", ln=True)
                if isinstance(metrics, dict):
                     for metric, value in metrics.items():
                          value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                          pdf.cell(200, 8, txt=f"    {metric}: {value_str}", ln=True)
                else:
                     pdf.cell(200, 8, txt=f"    Result: {metrics}", ln=True)
        else:
             pdf.cell(200, 8, txt=f"  Result: {tasks}", ln=True)

        pdf.ln(5)
    pdf.output(str(file_path))

# --- Registry ---
_SAVERS: Dict[str, Callable[[Dict[str, Any], Path], None]] = {
    "json": _save_json,
    "yaml": _save_yaml,
    "csv": _save_csv,
    "pickle": _save_pickle,
    "parquet": _save_parquet,
    "pdf": _save_pdf,
}

def save_results(
    results: Dict[str, Any],
    output_dir: str = "results",
    format: str = "json"
) -> None:
    """
    Saves benchmark results using a registry pattern.
    :param results: Dictionary containing benchmark results.
    :param output_dir: Directory to save the results.
    :param format: Format to save the results. Supported formats: json, yaml, csv, pickle, parquet, pdf.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    format_lower = format.lower()
    saver_func = _SAVERS.get(format_lower)

    if not saver_func:
        supported_formats = ", ".join(_SAVERS.keys())
        raise ValueError(f"Unsupported format: {format}. Choose from: {supported_formats}")

    file_extension = format_lower
    if format_lower == "pickle": file_extension = "pkl"

    file_path = output_dir_path / f"benchmark_results.{file_extension}"

    try:
        logger.info(f"Saving results to {file_path}...")
        saver_func(results, file_path)
        logger.info(f"Results successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save results in {format} format: {e}", exc_info=True)