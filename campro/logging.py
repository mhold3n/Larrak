"""Centralized logging configuration for Larrak project."""

from __future__ import annotations

import logging
from typing import Final


_PACKAGE_LOGGER_LEVEL: Final[int] = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger without altering global handlers.

    The repository policy forbids custom handlers at module level.
    """
    logger = logging.getLogger(name)
    # Do not configure handlers here; respect application-level config.
    logger.setLevel(_PACKAGE_LOGGER_LEVEL)
    return logger





