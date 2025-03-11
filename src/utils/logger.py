import logging
import logging.config
import sys

def setup_logger():
    """
    Sets up the logging configuration based on config file.
    """
    logging.config.fileConfig('config/logging.conf')
    logger = logging.getLogger('benchmark')
    return logger

logger = setup_logger()