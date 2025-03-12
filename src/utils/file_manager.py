import os
import json

def save_results(results, output_dir="results"):
    """
    Save benchmark results to a file in JSON format.
    
    :param results: The results dictionary to save.
    :param output_dir: The directory to save the results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_file = os.path.join(output_dir, "benchmark_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)