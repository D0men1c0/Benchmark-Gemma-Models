import yaml
import sys
sys.path.append("src")
from models.benchmark import BenchmarkRunner
from utils.logger import setup_logger

# Set up logger
logger = setup_logger()

def main():
    # Load configuration from the config file
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create and run the benchmark
    benchmark_runner = BenchmarkRunner(config)
    results = benchmark_runner.run()
    
    # Print results and save them
    logger.info(f"Benchmark results: {results}")

if __name__ == "__main__":
    main()