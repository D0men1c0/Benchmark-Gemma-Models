import os
import json
import yaml
import pickle
import csv
from utils.logger import setup_logger
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from fpdf import FPDF

logger = setup_logger()

def save_results(
    results: Dict[str, Any], 
    output_dir: str = "results", 
    format: str = "json"
) -> None:
    """
    Save benchmark results to a file in the specified format.

    Supported formats: json, yaml, csv, pickle, parquet, pdf

    :param results: The results dictionary to save.
    :param output_dir: The directory to save the results.
    :param format: The file format to use (json, yaml, csv, pickle, parquet, pdf).
    :raises ValueError: If an unsupported format is provided.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Determine the file extension and save logic based on the format
        if format == "json":
            file_path = Path(output_dir) / "benchmark_results.json"
            with open(file_path, "w") as f:
                json.dump(results, f, indent=4)
        
        elif format == "yaml":
            file_path = Path(output_dir) / "benchmark_results.yaml"
            with open(file_path, "w") as f:
                yaml.safe_dump(results, f, default_flow_style=False)
        
        elif format == "csv":
            file_path = Path(output_dir) / "benchmark_results.csv"
            # Flatten results for CSV
            flattened = []
            for model, tasks in results.items():
                for task, metrics in tasks.items():
                    row = {"model": model, "task": task}
                    row.update(metrics)
                    flattened.append(row)
            
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
        
        elif format == "pickle":
            file_path = Path(output_dir) / "benchmark_results.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(results, f)
        
        elif format == "parquet":
            file_path = Path(output_dir) / "benchmark_results.parquet"
            # Convert results to DataFrame
            flattened = []
            for model, tasks in results.items():
                for task, metrics in tasks.items():
                    row = {"model": model, "task": task}
                    row.update(metrics)
                    flattened.append(row)
            
            df = pd.DataFrame(flattened)
            df.to_parquet(file_path, engine="pyarrow")
        
        elif format == "pdf":
            file_path = Path(output_dir) / "benchmark_results.pdf"
            # Create a PDF report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Add title
            pdf.cell(200, 10, txt="Benchmark Results", ln=True, align="C")
            pdf.ln(10)
            
            # Add content
            for model, tasks in results.items():
                pdf.cell(200, 10, txt=f"Model: {model}", ln=True)
                for task, metrics in tasks.items():
                    pdf.cell(200, 10, txt=f"  Task: {task}", ln=True)
                    for metric, value in metrics.items():
                        pdf.cell(200, 10, txt=f"    {metric}: {value}", ln=True)
                pdf.ln(5)
            
            pdf.output(file_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Choose from: json, yaml, csv, pickle, parquet, pdf")

        logger.info(f"Results successfully saved to {file_path}")
    
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise