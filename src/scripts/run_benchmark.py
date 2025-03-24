import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from benchmark.benchmark_loader import BenchmarkRunner

logger = setup_logger()

def validate_config(config: Dict[str, Any]) -> bool:
    """Basic configuration validation"""
    required_keys = {"models", "tasks"}
    if not required_keys.issubset(config.keys()):
        missing = required_keys - config.keys()
        raise ValueError(f"Missing required config keys: {missing}")
    return True

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            validate_config(config)
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ML Benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "prove123.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    return parser.parse_args()

def main() -> None:
    """Main execution flow"""
    args = parse_args()
    
    # Initialize logging
    logger = setup_logger()
    
    try:
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        
        logger.info("Initializing benchmark runner")
        benchmark_runner = BenchmarkRunner(config)
        
        logger.info("Starting benchmark execution")
        results = benchmark_runner.run()
        
        logger.info("Benchmark completed successfully")
        logger.debug(f"Full results: {results}")
        
    except Exception as e:
        logger.critical(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()