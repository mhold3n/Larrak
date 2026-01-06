"""Custom logging handlers for structured logging and session management.

This module provides specialized handlers for:
- Session-scoped log aggregation
- Level-based file routing
- Metrics extraction from log messages
- Disk space management
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar


class StructuredFormatter(logging.Formatter):
    """JSON formatter with contextual metadata for machine-readable logs."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with metadata."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=datetime.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add contextual metadata if available
        if hasattr(record, "run_id"):
            log_data["run_id"] = record.run_id
        if hasattr(record, "module_id"):
            log_data["module_id"] = record.module_id
        if hasattr(record, "iteration"):
            log_data["iteration"] = record.iteration
        if hasattr(record, "objective"):
            log_data["objective"] = record.objective

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]:
                    log_data[key] = value

        return json.dumps(log_data)


class LevelBasedFileHandler(logging.Handler):
    """Routes log records to different files based on log level."""

    def __init__(self, base_dir: Path):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each level
        self.level_dirs = {
            logging.ERROR: self.base_dir / "errors",
            logging.WARNING: self.base_dir / "warnings",
            logging.INFO: self.base_dir / "info",
        }

        for level_dir in self.level_dirs.values():
            level_dir.mkdir(parents=True, exist_ok=True)

        # File handlers for each level (daily rotation)
        self.handlers: dict[int, logging.FileHandler] = {}
        self._update_handlers()

    def _update_handlers(self) -> None:
        """Update file handlers with current date."""
        today = datetime.now(datetime.UTC).strftime("%Y-%m-%d")

        for level, level_dir in self.level_dirs.items():
            if level not in self.handlers or not self._is_current_date(self.handlers[level]):
                log_file = level_dir / f"{today}.jsonl"
                handler = logging.FileHandler(log_file, mode="a")
                handler.setFormatter(StructuredFormatter())
                self.handlers[level] = handler

    def _is_current_date(self, handler: logging.FileHandler) -> bool:
        """Check if handler's file is for current date."""
        if not hasattr(handler, "baseFilename"):
            return False
        filename = Path(handler.baseFilename).stem
        today = datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        return filename == today

    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to appropriate level-based file."""
        self._update_handlers()

        # Route to appropriate handler
        if record.levelno >= logging.ERROR:
            handler = self.handlers.get(logging.ERROR)
        elif record.levelno >= logging.WARNING:
            handler = self.handlers.get(logging.WARNING)
        elif record.levelno >= logging.INFO:
            handler = self.handlers.get(logging.INFO)
        else:
            # DEBUG logs not written to level-based files
            return

        if handler:
            handler.emit(record)


class SessionLogHandler(logging.Handler):
    """Manages per-session log directory with full retention."""

    def __init__(self, session_id: str, base_dir: Path):
        super().__init__()
        self.session_id = session_id
        self.session_dir = Path(base_dir) / "sessions" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Full log (all levels)
        self.full_log_handler = logging.FileHandler(self.session_dir / "full.log")
        self.full_log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        # Debug log (rotating, last 100MB)
        self.debug_log_handler = RotatingFileHandler(
            self.session_dir / "debug.log", maxBytes=100 * 1024 * 1024, backupCount=1
        )
        self.debug_log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        # Metadata file
        self._write_metadata()

    def _write_metadata(self) -> None:
        """Write session metadata file."""
        metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now(datetime.UTC).isoformat(),
            "status": "running",
        }
        metadata_file = self.session_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to session log files."""
        # Write all levels to full log
        self.full_log_handler.emit(record)

        # Write debug to rotating debug log
        if record.levelno == logging.DEBUG:
            self.debug_log_handler.emit(record)

    def close_session(self, status: str = "completed") -> None:
        """Mark session as closed in metadata."""
        metadata_file = self.session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            metadata["end_time"] = datetime.now(timezone.utc).isoformat()
            metadata["status"] = status
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        self.full_log_handler.close()
        self.debug_log_handler.close()


class MetricsExtractor(logging.Handler):
    """Extracts structured metrics from log messages."""

    # Patterns to extract metrics from log messages
    METRIC_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "objective": re.compile(r"(?:objective|obj|best)[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"),
        "iteration": re.compile(r"(?:iteration|iter)[:\s]+(\d+)"),
        "inf_pr": re.compile(r"inf_pr[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"),
        "inf_du": re.compile(r"inf_du[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"),
        "convergence_rate": re.compile(
            r"convergence[_\s]+rate[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
        ),
    }

    def __init__(self, metrics_file: Path):
        super().__init__()
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        """Extract metrics from log message and write to JSONL."""
        message = record.getMessage()
        metrics: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=datetime.UTC).isoformat(),
            "logger": record.name,
        }

        # Extract metrics using patterns
        for metric_name, pattern in self.METRIC_PATTERNS.items():
            match = pattern.search(message)
            if match:
                try:
                    value = float(match.group(1))
                    metrics[metric_name] = value
                except ValueError:
                    pass

        # Only write if we extracted at least one metric
        if len(metrics) > 2:  # More than just timestamp and logger
            # Add run_id if available
            if hasattr(record, "run_id"):
                metrics["run_id"] = record.run_id

            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")


class DiskSpaceManager:
    """Monitors and manages log directory disk usage."""

    def __init__(self, log_base_dir: Path, max_size_gb: float = 10.0):
        self.log_base_dir = Path(log_base_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

    def get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for path in directory.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        except (PermissionError, FileNotFoundError):
            pass
        return total

    def check_and_enforce(self) -> dict[str, Any]:
        """Check disk usage and enforce limits if necessary."""
        current_size = self.get_directory_size(self.log_base_dir)
        size_gb = current_size / (1024 * 1024 * 1024)

        status = {
            "current_size_gb": round(size_gb, 2),
            "max_size_gb": self.max_size_bytes / (1024 * 1024 * 1024),
            "usage_percent": round((current_size / self.max_size_bytes) * 100, 1),
            "action_taken": None,
        }

        if current_size > self.max_size_bytes:
            # Trigger cleanup (this would call log_cleanup.py)
            status["action_taken"] = "cleanup_triggered"
            # Import and run cleanup here (or trigger via subprocess)
            # For now, just log a warning
            logging.warning(
                f"Log directory exceeds limit: {size_gb:.2f}GB / "
                f"{self.max_size_bytes / (1024**3):.2f}GB. Cleanup recommended."
            )

        return status
