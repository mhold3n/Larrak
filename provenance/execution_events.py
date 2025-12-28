"""Execution events for live dashboard tracking.

Defines event types and emitter functions for broadcasting
orchestrator state to connected dashboards.

Usage:
    from provenance.execution_events import emit_event, EventType

    emit_event(EventType.MODULE_START, module="CEM")
    emit_event(EventType.TOOL_CALL, module="SOL", tool="ipopt")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from campro.logging import get_logger

log = get_logger(__name__)


class EventType(Enum):
    """Types of execution events."""

    # Module lifecycle
    MODULE_START = "module_start"
    MODULE_END = "module_end"

    # Tool execution
    TOOL_CALL = "tool_call"
    TOOL_COMPLETE = "tool_complete"

    # Data flow
    FLOW_START = "flow_start"  # Data moving between modules
    FLOW_END = "flow_end"

    # Optimization step
    STEP_START = "step_start"
    STEP_END = "step_end"

    # Errors
    ERROR = "error"
    WARNING = "warning"
    LOG = "log"

    # Run lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"


@dataclass
class ExecutionEvent:
    """A single execution event."""

    type: EventType
    module: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def to_json(self) -> str:
        """Convert to JSON string for WebSocket transmission."""
        data = asdict(self)
        data["type"] = self.type.value
        return json.dumps(data)


# Global event listeners
_listeners: list[Callable[[ExecutionEvent], None]] = []
_event_history: list[ExecutionEvent] = []
_current_run_id: str | None = None
_module_start_times: dict[str, float] = {}


def add_listener(listener: Callable[[ExecutionEvent], None]) -> None:
    """Add event listener (e.g., WebSocket broadcaster)."""
    _listeners.append(listener)


def remove_listener(listener: Callable[[ExecutionEvent], None]) -> None:
    """Remove event listener."""
    if listener in _listeners:
        _listeners.remove(listener)


def get_history(limit: int = 100) -> list[ExecutionEvent]:
    """Get recent event history for new connections."""
    return _event_history[-limit:]


def set_run_id(run_id: str) -> None:
    """Set current run ID for all events."""
    global _current_run_id
    _current_run_id = run_id


def emit_event(
    event_type: EventType,
    module: str,
    tool: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionEvent:
    """Emit an execution event to all listeners.

    Args:
        event_type: Type of event
        module: Module ID (CEM, SUR, SOL, etc.)
        tool: Optional tool name (casadi, ipopt, etc.)
        metadata: Optional additional data

    Returns:
        The created event
    """
    # Calculate duration for END events
    duration_ms = None
    if event_type in (EventType.MODULE_END, EventType.TOOL_COMPLETE):
        key = f"{module}:{tool}" if tool else module
        if key in _module_start_times:
            duration_ms = (time.time() - _module_start_times[key]) * 1000
            del _module_start_times[key]

    # Track start times
    if event_type in (EventType.MODULE_START, EventType.TOOL_CALL):
        key = f"{module}:{tool}" if tool else module
        _module_start_times[key] = time.time()

    event = ExecutionEvent(
        type=event_type,
        module=module,
        tool=tool,
        duration_ms=duration_ms,
        metadata=metadata or {},
        run_id=_current_run_id,
    )

    # Store in history
    _event_history.append(event)
    if len(_event_history) > 1000:
        _event_history.pop(0)

    # Notify listeners
    for listener in _listeners:
        try:
            listener(event)
        except Exception as e:
            log.warning(f"Event listener error: {e}")

    log.debug(f"Event: {event_type.value} {module}" + (f" ({tool})" if tool else ""))
    return event


# Convenience functions
def module_start(module: str, **metadata: Any) -> ExecutionEvent:
    """Emit MODULE_START event."""
    return emit_event(EventType.MODULE_START, module, metadata=metadata)


def module_end(module: str, **metadata: Any) -> ExecutionEvent:
    """Emit MODULE_END event."""
    return emit_event(EventType.MODULE_END, module, metadata=metadata)


def tool_call(module: str, tool: str, **metadata: Any) -> ExecutionEvent:
    """Emit TOOL_CALL event."""
    return emit_event(EventType.TOOL_CALL, module, tool=tool, metadata=metadata)


def tool_complete(module: str, tool: str, **metadata: Any) -> ExecutionEvent:
    """Emit TOOL_COMPLETE event."""
    return emit_event(EventType.TOOL_COMPLETE, module, tool=tool, metadata=metadata)


def flow_start(source: str, target: str, label: str = "") -> ExecutionEvent:
    """Emit FLOW_START event for data moving between modules."""
    return emit_event(
        EventType.FLOW_START,
        source,
        metadata={"target": target, "label": label},
    )


def flow_end(source: str, target: str) -> ExecutionEvent:
    """Emit FLOW_END event."""
    return emit_event(EventType.FLOW_END, source, metadata={"target": target})


def step_start(iteration: int, **metadata: Any) -> ExecutionEvent:
    """Emit STEP_START event."""
    return emit_event(EventType.STEP_START, "ORCH", metadata={"iteration": iteration, **metadata})


def step_end(iteration: int, **metadata: Any) -> ExecutionEvent:
    """Emit STEP_END event."""
    return emit_event(EventType.STEP_END, "ORCH", metadata={"iteration": iteration, **metadata})


def error(module: str, message: str, **metadata: Any) -> ExecutionEvent:
    """Emit ERROR event."""
    return emit_event(EventType.ERROR, module, metadata={"message": message, **metadata})


def warning(module: str, message: str, **metadata: Any) -> ExecutionEvent:
    """Emit WARNING event."""
    return emit_event(EventType.WARNING, module, metadata={"message": message, **metadata})


def log_message(module: str, message: str, level: str = "INFO", **metadata: Any) -> ExecutionEvent:
    """Emit generic LOG event."""
    return emit_event(
        EventType.LOG, module, metadata={"message": message, "level": level, **metadata}
    )
