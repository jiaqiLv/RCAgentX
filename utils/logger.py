"""
Logging Configuration for AIOps System

Centralized logging setup with support for multiple outputs,
structured logging, and configurable log levels.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance.

    Creates or retrieves a logger with the specified name and
    configures it with console and optional file output.

    Args:
        name (str): Logger name (typically __name__)
        level (int): Logging level (default: INFO)
        log_file (Optional[str]): Path to log file. If None, only
            console output is enabled.
        format_string (Optional[str]): Custom format string. If None,
            uses default format with timestamp and level.

    Returns:
        logging.Logger: Configured logger instance

    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("Application started")
        logger.error("Something went wrong")
        ```

    Default format:
        YYYY-MM-DD HH:MM:SS - LEVEL - name - MESSAGE
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Default format string
    if format_string is None:
        format_string = (
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AIOpsLogger:
    """
    Centralized logger for the AIOps system.

    Provides pre-configured loggers for different components
    with consistent formatting and output.

    Example:
        ```python
        # Get component-specific logger
        logger = AIOpsLogger.get("supervisor")
        logger.info("Workflow started")

        # Get root logger
        root_logger = AIOpsLogger.get_root()
        ```
    """

    _loggers = {}
    _default_level = logging.INFO
    _log_dir: Optional[str] = None

    @classmethod
    def get(
        cls,
        component: str,
        level: Optional[int] = None
    ) -> logging.Logger:
        """
        Get a logger for a specific component.

        Args:
            component (str): Component name (supervisor, observability, etc.)
            level (Optional[int]): Custom log level

        Returns:
            logging.Logger: Component-specific logger
        """
        if component not in cls._loggers:
            logger = get_logger(
                name=f"aiops.{component}",
                level=level or cls._default_level,
                log_file=cls._get_log_file(component)
            )
            cls._loggers[component] = logger

        return cls._loggers[component]

    @classmethod
    def get_root(cls) -> logging.Logger:
        """
        Get the root AIOps logger.

        Returns:
            logging.Logger: Root logger
        """
        return cls.get("root")

    @classmethod
    def set_level(cls, level: int):
        """
        Set default log level for all loggers.

        Args:
            level (int): Logging level
        """
        cls._default_level = level

        # Update existing loggers
        for logger in cls._loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

    @classmethod
    def set_log_dir(cls, log_dir: str):
        """
        Set the log directory for file output.

        Args:
            log_dir (str): Path to log directory
        """
        cls._log_dir = log_dir

    @classmethod
    def _get_log_file(cls, component: str) -> Optional[str]:
        """
        Get log file path for a component.

        Args:
            component (str): Component name

        Returns:
            Optional[str]: Log file path or None if not configured
        """
        if not cls._log_dir:
            return None

        date_str = datetime.now().strftime("%Y%m%d")
        return f"{cls._log_dir}/aiops_{component}_{date_str}.log"


# Convenient function for quick logging setup
def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None
):
    """
    Setup logging for the entire AIOps system.

    Args:
        level (int): Default log level
        log_dir (Optional[str]): Directory for log files

    Example:
        ```python
        setup_logging(level=logging.DEBUG, log_dir="./logs")
        ```
    """
    AIOpsLogger.set_level(level)
    AIOpsLogger.set_log_dir(log_dir)

    # Log setup confirmation
    logger = AIOpsLogger.get_root()
    logger.info(f"Logging initialized with level {logging.getLevelName(level)}")
    if log_dir:
        logger.info(f"Log files will be written to {log_dir}")
