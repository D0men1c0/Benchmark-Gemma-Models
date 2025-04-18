import logging
import sys

# Cache to avoid duplicate handlers
_loggers = {}

def setup_logger(name: str = 'benchmark', level: int = logging.INFO) -> logging.Logger:
    """
    Set up the logger with a basic configuration. Avoids duplicate handlers.

    :param name: Name for the logger.
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    :return: Configured logger instance.
    """
    # If the logger already exists and is configured, return it
    if name in _loggers:
        return _loggers[name]

    # Otherwise, create and configure a new logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent messages from being passed to the root logger

    # Configure the handler only if it doesn't already have one of the same type
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout) # Use sys.stdout explicitly
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Add the configured logger to the cache
    _loggers[name] = logger
    return logger

# Convenience function to set the global log level (optional)
def set_global_log_level(level: int):
    """Sets the level for all cached loggers."""
    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)