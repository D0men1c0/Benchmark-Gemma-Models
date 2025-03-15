import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from models.benchmark_loader import BenchmarkRunner
from utils.logger import setup_logger

# Set up logger
logger = setup_logger()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "config", "benchmark_config(1).yaml")

def main():
    # Load configuration from the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create and run the benchmark
    benchmark_runner = BenchmarkRunner(config)
    results = benchmark_runner.run()
    
    # Print results and save them
    logger.info(f"Benchmark results: {results}")

if __name__ == "__main__":
    main()