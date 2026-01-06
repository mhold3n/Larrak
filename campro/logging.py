"""Centralized logging configuration for Larrak project.

This module provides:
- Standard logger factory (get_logger) for module-level loggers
- Structured logging configuration for optimization sessions
- Session context management for log aggregation
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from campro.logging_handlers import LevelBasedFileHandler, MetricsExtractor, SessionLogHandler

# Allow environment override without touching handlers
_LEVEL_NAME: Final[str] = os.getenv("CAMPRO_LOG_LEVEL", "INFO").upper()
_PACKAGE_LOGGER_LEVEL: Final[int] = getattr(logging, _LEVEL_NAME, logging.INFO)

# Global session context
_current_session_id: str | None = None
_session_handlers: list[logging.Handler] = []


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger without altering global handlers.

    The repository policy forbids custom handlers at module level.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    # Do not configure handlers here; respect application-level config.
    logger.setLevel(_PACKAGE_LOGGER_LEVEL)
    return logger


def get_log_base_dir() -> Path:
    """Get base directory for logs.

    Returns:
        Path to logs directory (defaults to PROJECT_ROOT/logs)
    """
    log_dir = os.getenv("LARRAK_LOG_DIR")
    if log_dir:
        return Path(log_dir)

    # Default to project root / logs
    project_root = Path(__file__).parent.parent
    return project_root / "logs"


def configure_root_logging(
    level: int = logging.INFO,
    enable_structured: bool = True,
    enable_metrics: bool = True,
) -> None:
    """Configure root logger with structured logging and handlers.

    This should be called once at application startup.

    Args:
        level: Minimum log level
        enable_structured: Enable level-based structured file logging
        enable_metrics: Enable metrics extraction to JSONL
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(console_handler)

    log_base_dir = get_log_base_dir()

    # Level-based file handler (structured JSON)
    if enable_structured:
        structured_dir = log_base_dir / "structured"
        level_handler = LevelBasedFileHandler(structured_dir)
        level_handler.setLevel(logging.INFO)
        root_logger.addHandler(level_handler)

    # Metrics extractor
    if enable_metrics:
        metrics_dir = log_base_dir / "metrics"
        metrics_file = metrics_dir / "optimization_metrics.jsonl"
        metrics_handler = MetricsExtractor(metrics_file)
        metrics_handler.setLevel(logging.INFO)
        root_logger.addHandler(metrics_handler)


@contextmanager
def session_logging(
    session_id: str,
    params: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Context manager for session-scoped logging.

    Creates a session log directory and routes all logs during the context
    to both session files and global handlers.

    Args:
        session_id: Unique session identifier (e.g., run_id)
        params: Optional session parameters to log

    Yields:
        session_id for reference

    Example:
        with session_logging("run_abc123", params={"batch_size": 10}):
            logger.info("Starting optimization")
            # All logs written to logs/sessions/run_abc123/full.log
    """
    global _current_session_id
    global _session_handlers

    _current_session_id = session_id
    log_base_dir = get_log_base_dir()

    # Create session handler
    session_handler = SessionLogHandler(session_id, log_base_dir)
    session_handler.setLevel(logging.DEBUG)  # Capture all levels for session

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(session_handler)
    _session_handlers.append(session_handler)

    # Log session start
    logger = get_logger("campro.logging")
    logger.info(f"Session {session_id} started")
    if params:
        logger.info(f"Session parameters: {params}")

    try:
        yield session_id
    except Exception as e:
        logger.error(f"Session {session_id} failed with exception: {e}")
        session_handler.close_session(status="failed")
        raise
    else:
        logger.info(f"Session {session_id} completed successfully")
        session_handler.close_session(status="completed")
    finally:
        # Remove session handler
        root_logger.removeHandler(session_handler)
        _session_handlers.remove(session_handler)
        _current_session_id = None


def add_session_context(record: logging.LogRecord) -> logging.LogRecord:
    """Add current session context to log record.

    This can be used as a filter to automatically inject session_id
    into all log records during a session.

    Args:
        record: Log record to modify

    Returns:
        Modified record with session context
    """
    if _current_session_id:
        record.run_id = _current_session_id
    return record


class SessionContextFilter(logging.Filter):
    """Logging filter that adds session context to records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add session context if available."""
        if _current_session_id and not hasattr(record, "run_id"):
            record.run_id = _current_session_id
        return True
