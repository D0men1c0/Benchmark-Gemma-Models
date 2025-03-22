import logging
import inspect

def setup_logger():
    """
    Set up the logger with a basic configuration.
    The logger name is dynamically set to the name of the class that calls this function.
    """
    caller_frame = inspect.currentframe().f_back
    caller_class_name = caller_frame.f_locals.get('self', None).__class__.__name__
    logger = logging.getLogger(caller_class_name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger