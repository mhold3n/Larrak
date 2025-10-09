"""Centralized logging configuration for Larrak project."""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    Parameters
    ----------
    name : str
        Module name (typically __name__)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
        
    Example
    -------
    >>> from campro.logging import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Module initialized")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if logs directory exists)
        logs_dir = Path("logs")
        if logs_dir.exists():
            file_handler = logging.FileHandler(
                logs_dir / f"{name.replace('.', '_')}.log",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        # Ensure file handler is present if logs dir exists and not already added
        logs_dir = Path("logs")
        if logs_dir.exists() and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            file_handler = logging.FileHandler(
                logs_dir / f"{name.replace('.', '_')}.log",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger





