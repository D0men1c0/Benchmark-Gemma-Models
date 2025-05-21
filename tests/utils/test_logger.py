import logging
import pytest
from pathlib import Path

from src.utils.logger import LoggerManager, setup_logger #

# Simple Tests
def test_setup_logger_returns_logger_instance():
    """Test that setup_logger returns a logging.Logger instance."""
    logger = setup_logger("test_simple_logger", console=False, to_file=False)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_simple_logger"

def test_logger_manager_singleton_behavior(tmp_path):
    """Test that LoggerManager behaves like a singleton for file handling."""
    manager1 = LoggerManager(log_dir=str(tmp_path / "logs1"))
    manager2 = LoggerManager(log_dir=str(tmp_path / "logs2")) # This dir should be ignored
    assert manager1 is manager2
    assert manager1.log_dir == tmp_path / "logs1" # Initial log_dir is kept


def test_logger_writes_to_file(tmp_path):
    """Test that a logger writes messages to the specified file."""
    log_file_name = "test_file_log.log"
    logger = setup_logger("test_file_writer", log_dir=str(tmp_path), file_name=log_file_name, console=False, to_file=True)
    
    test_message = "This message should be in the file."
    logger.warning(test_message)

    # Important: Ensure logs are flushed. In real scenarios, handlers might buffer.
    # For testing, explicitly close handlers or use a more direct way to check file content.
    # Here, we assume default buffering isn't an issue for a single message or use LoggerManager's file_handler.
    
    log_file_path = tmp_path / log_file_name
    assert log_file_path.exists()
    
    # Force close handlers to ensure flush
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    # If using the singleton, the file_handler might persist.
    # For robust testing, you might need to reset/reinitialize the LoggerManager instance
    # or directly interact with its file_handler.

    with open(log_file_path, "r") as f:
        content = f.read()
    assert test_message in content
    assert "test_file_writer" in content # Check logger name
    assert "WARNING" in content # Check level


@pytest.fixture(autouse=True)
def reset_logger_manager_instance():
    """Fixture to reset the LoggerManager singleton before and after each test."""
    LoggerManager._instance = None # Reset before test
    yield
    LoggerManager._instance = None # Reset after test