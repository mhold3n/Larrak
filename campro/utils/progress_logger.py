"""Structured progress logging utility for optimization routines."""

from __future__ import annotations

import sys
import time
from typing import Any

from campro.logging import get_logger

log = get_logger(__name__)


class ProgressLogger:
    """Structured progress logger for optimization phases."""

    def __init__(self, phase: str, flush_immediately: bool = True):
        """Initialize progress logger for a specific phase.

        Parameters
        ----------
        phase : str
            Phase name (e.g., "PRIMARY", "SECONDARY", "TERTIARY")
        flush_immediately : bool
            If True, flush output immediately for real-time visibility
        """
        self.phase = phase.upper()
        self.flush_immediately = flush_immediately
        self.start_time: float | None = None
        self.step_times: list[tuple[str, float]] = []

    def _format_message(self, step: str, message: str, status: str | None = None) -> str:
        """Format a progress message with consistent structure."""
        prefix = f"[{self.phase}]"
        if status:
            status_symbol = {
                "start": "⏳",
                "success": "✓",
                "error": "✗",
                "info": "ℹ",
                "warning": "⚠",
            }.get(status, "")
            if status_symbol:
                prefix = f"{prefix} {status_symbol}"
        return f"{prefix} {step}: {message}"

    def _output(self, message: str, level: str = "info"):
        """Output message with optional immediate flush."""
        formatted_msg = message
        if level == "info":
            print(formatted_msg, file=sys.stderr, flush=self.flush_immediately)
            log.info(formatted_msg)
        elif level == "warning":
            print(formatted_msg, file=sys.stderr, flush=self.flush_immediately)
            log.warning(formatted_msg)
        elif level == "error":
            print(formatted_msg, file=sys.stderr, flush=self.flush_immediately)
            log.error(formatted_msg)
        elif level == "debug":
            print(formatted_msg, file=sys.stderr, flush=self.flush_immediately)
            log.debug(formatted_msg)

    def start_phase(self, total_steps: int | None = None) -> None:
        """Mark the start of a phase."""
        self.start_time = time.time()
        if total_steps:
            msg = self._format_message(
                "Phase Start",
                f"Starting optimization ({total_steps} steps total)",
                "start",
            )
        else:
            msg = self._format_message("Phase Start", "Starting optimization", "start")
        self._output(msg)

    def step(self, step_num: int | None, total_steps: int | None, description: str, status: str | None = None) -> None:
        """Log a step with optional numbering."""
        step_start = time.time()
        if step_num is not None and total_steps is not None:
            step_label = f"Step {step_num}/{total_steps}"
        elif step_num is not None:
            step_label = f"Step {step_num}"
        else:
            step_label = "Step"
        msg = self._format_message(step_label, description, status)
        self._output(msg)
        self.step_times.append((description, step_start))

    def step_complete(self, description: str, elapsed: float | None = None) -> None:
        """Mark a step as complete with timing."""
        if elapsed is None:
            # Find the last step with this description
            for desc, step_time in reversed(self.step_times):
                if desc == description:
                    elapsed = time.time() - step_time
                    break
        if elapsed is not None:
            msg = self._format_message(
                "Complete",
                f"{description} completed in {elapsed:.3f}s",
                "success",
            )
        else:
            msg = self._format_message("Complete", f"{description} completed", "success")
        self._output(msg)

    def info(self, message: str) -> None:
        """Log an informational message."""
        msg = self._format_message("Info", message, "info")
        self._output(msg)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        msg = self._format_message("Warning", message, "warning")
        self._output(msg, level="warning")

    def error(self, message: str) -> None:
        """Log an error message."""
        msg = self._format_message("Error", message, "error")
        self._output(msg, level="error")

    def complete_phase(self, success: bool = True) -> None:
        """Mark the phase as complete with total elapsed time."""
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if success:
            msg = self._format_message(
                "Phase Complete",
                f"Optimization completed successfully in {elapsed:.3f}s",
                "success",
            )
        else:
            msg = self._format_message(
                "Phase Complete",
                f"Optimization failed after {elapsed:.3f}s",
                "error",
            )
        self._output(msg)

    def separator(self) -> None:
        """Print a visual separator."""
        sep = f"[{self.phase}] {'=' * 60}"
        print(sep, file=sys.stderr, flush=self.flush_immediately)

