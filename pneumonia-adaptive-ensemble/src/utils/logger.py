"""Logging utility configured for research logging."""
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pneumonia_ensemble",
    log_dir: Optional[str] = "results/logs",
    log_filename: Optional[str] = "experiment.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger that writes to both console and log file.

    Args:
        name: Name of the logger.
        log_dir: Directory where log file will be saved.
        log_filename: Name of the log file.
        level: Logging severity level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File Handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = Path(log_dir) / (log_filename or "run.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
