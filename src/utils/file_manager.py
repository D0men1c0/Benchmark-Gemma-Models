import os
import json
from typing import Any, Dict

def save_results(results: Dict[str, Any], output_path: str):
    """
    Save benchmark results to a file.

    :param results: The benchmark results to save.
    :param output_path: The path where to save the results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)