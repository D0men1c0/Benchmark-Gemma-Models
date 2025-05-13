import argparse
import sys
import logging
from pathlib import Path
import yaml
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from benchmark.benchmark_loader import BenchmarkRunner
from config_models import BenchmarkConfig
from pydantic import ValidationError

def load_config(config_path: Path) -> BenchmarkConfig:
    """
    Load and validate configuration file using Pydantic.
    :param config_path: Path to the YAML configuration file.
    :return: A validated BenchmarkConfig object.
    """
    logger = setup_logger(__name__)
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
             raise ValueError("Configuration file is empty or invalid.")
        # Use model_validate for Pydantic v2+
        config = BenchmarkConfig.model_validate(config_data)
        logger.info(f"Configuration loaded and validated successfully from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        raise
    except ValidationError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        raise
    except ValueError as e: # Catch specific value errors like empty file
        logger.error(f"Error loading configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration: {e}")
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ML Benchmark")
    # Adjust default config path if needed, assuming execution from project root
    default_config_path = Path("src") / "config" / "benchmark_config.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=f"Path to the YAML configuration file (default: {default_config_path})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory specified in the config file.",
    )
    return parser.parse_args()

def main() -> None:
    """Main execution flow"""
    args = parse_args()

    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logger = setup_logger(__name__)

    try:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config) # Get the validated BenchmarkConfig object

        # Override config with CLI arguments if provided
        if args.output_dir:
            logger.info(f"Overriding output directory with CLI argument: {args.output_dir}")
            config.general.output_dir = args.output_dir
            # Check if reporting config exists before trying to set its output_dir
            if config.reporting:
                 config.reporting.output_dir = args.output_dir

        logger.info("Initializing benchmark runner")
        benchmark_runner = BenchmarkRunner(config)

        logger.info("Starting benchmark execution")
        results = benchmark_runner.run()

        logger.info("Benchmark completed successfully")
        if logger.isEnabledFor(logging.DEBUG):
             import json
             try:
                 # Use Pydantic's json encoder for robust serialization
                 results_json = json.dumps(results, indent=2, default=str) # Add default=str for non-serializable types
                 logger.debug(f"Full results:\n{results_json}")
             except TypeError as e:
                 logger.debug(f"Could not serialize full results to JSON: {e}")

    except FileNotFoundError:
        logger.critical(f"Configuration file not found. Please check the path: {args.config}")
        sys.exit(1)
    except (yaml.YAMLError, ValidationError, ValueError) as e:
        logger.critical(f"Failed to load or validate the configuration file ({type(e).__name__}): {e}")
        sys.exit(1)
    except ImportError as e:
         logger.critical(f"A required library is missing: {e}. Please check your installation and dependencies.")
         sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the benchmark execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()