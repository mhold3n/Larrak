"""Centralized logging configuration for Larrak project."""

from __future__ import annotations

import logging
import os
from typing import Final

# Allow environment override without touching handlers
_LEVEL_NAME: Final[str] = os.getenv("CAMPRO_LOG_LEVEL", "INFO").upper()
_PACKAGE_LOGGER_LEVEL: Final[int] = getattr(logging, _LEVEL_NAME, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger without altering global handlers.

    The repository policy forbids custom handlers at module level.
    """
    logger = logging.getLogger(name)
    # Do not configure handlers here; respect application-level config.
    logger.setLevel(_PACKAGE_LOGGER_LEVEL)
    return logger





