import sys
from models.benchmark import BenchmarkRunner
from src.utils.logger import logger
from src.utils.file_manager import save_results
import yaml

def load_config(config_file: str) -> dict:
    """
    Load configuration from YAML file.
    
    :param config_file: Path to the YAML configuration file.
    :return: Configuration dictionary.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main(config_file: str):
    """
    Main function to run the benchmark.
    
    :param config_file: Path to the configuration file.
    """
    config = load_config(config_file)
    logger.info(f"Loaded config from {config_file}")
    
    benchmark_runner = BenchmarkRunner(
        model_configs=config['models'],
        task_configs=config['tasks'],
        evaluation_params=config['evaluation']
    )
    
    results = benchmark_runner.run()
    logger.info("Benchmark completed. Saving results.")
    
    save_results(results, config['general']['output_dir'] + '/benchmark_results.json')

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/config.yaml'
    main(config_file)