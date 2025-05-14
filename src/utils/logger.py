import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

class LoggerManager:
    """
    Singleton manager to configure and retrieve loggers with optional
    console and shared file handlers.
    """
    _instance: Optional['LoggerManager'] = None

    def __new__(cls, log_dir: str = 'run_logs', file_name: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(log_dir, file_name)
        return cls._instance

    def _init(self, log_dir: str, file_name: Optional[str]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = file_name or f'benchmark_{timestamp}.log'
        self.file_path = self.log_dir / safe_name
        self._file_handler = self._create_file_handler()
        self._loggers: Dict[str, logging.Logger] = {}

    def _create_file_handler(self) -> logging.FileHandler:
        fh = logging.FileHandler(self.file_path, mode='a')
        fh.setFormatter(
            logging.Formatter(
               '%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
            )
        )
        return fh

    def get_logger(
        self,
        name: str,
        level: int = logging.INFO,
        console: bool = True,
        to_file: bool = True
    ) -> logging.Logger:
        """
        Return a logger configured with optional console and shared file handler.

        :param name: Logger name.
        :param level: Logging level.
        :param console: Attach a console handler if True.
        :param to_file: Attach shared file handler if True.
        :return: Configured logger.
        """
        if name in self._loggers:
            logger = self._loggers[name]
            logger.setLevel(level)
            return logger

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(ch)

        if to_file:
            self._file_handler.setLevel(level)
            logger.addHandler(self._file_handler)

        self._loggers[name] = logger
        return logger

    def set_global_level(self, level: int) -> None:
        """
        Update level for all managed loggers and shared file handler.

        :param level: New logging level.
        """
        for logger in self._loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
        self._file_handler.setLevel(level)


def setup_logger(
    name: str = 'benchmark_default',
    level: int = logging.INFO,
    console: bool = True,
    to_file: bool = True,
    log_dir: str = 'run_logs',
    file_name: Optional[str] = None
) -> logging.Logger:
    """
    Convenience function to get a configured logger via LoggerManager.

    :param name: Logger name.
    :param level: Logging level.
    :param console: Enable console handler.
    :param to_file: Enable shared file handler.
    :param log_dir: Directory for shared log file.
    :param file_name: Specific file name for shared log file.
    :return: Configured logger.
    """
    manager = LoggerManager(log_dir=log_dir, file_name=file_name)
    return manager.get_logger(name=name, level=level, console=console, to_file=to_file)