import logging

def setup_logger():
    """
    Set up the logger with a basic configuration.
    """
    logger = logging.getLogger("BenchmarkLogger")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger