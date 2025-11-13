from __future__ import annotations

import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, TextIO

import logging


_DEFAULT_DEBUG_ENV = "CAMPRO_DEBUG"
_FALSEY = {"0", "false", "no", "off"}
_WARNING_LEVELS = {"WARNING", "ERROR", "CRITICAL"}


@dataclass
class _ReporterState:
    indent: int
    indent_unit: str
    flush: bool
    show_debug: bool
    stream_out: TextIO
    stream_err: TextIO
    logger: Optional[logging.Logger]


class StructuredReporter:
    """Structured console reporter with hierarchical context support."""

    def __init__(
        self,
        *,
        context: Optional[str] = None,
        contexts: Tuple[str, ...] | None = None,
        indent: str = "  ",
        force_debug: Optional[bool] = None,
        show_debug: Optional[bool] = None,
        debug_env: Optional[str] = _DEFAULT_DEBUG_ENV,
        stream_out: TextIO = sys.stdout,
        stream_err: TextIO = sys.stderr,
        flush: bool = True,
        logger: Optional[logging.Logger] = None,
        state: _ReporterState | None = None,
    ) -> None:
        if contexts is not None:
            self._contexts: Tuple[str, ...] = tuple(contexts)
        elif context:
            self._contexts = (context,)
        else:
            self._contexts = ()

        if state is None:
            resolved_debug = self._resolve_debug(force_debug, show_debug, debug_env)
            self._state = _ReporterState(
                indent=0,
                indent_unit=indent,
                flush=flush,
                show_debug=resolved_debug,
                stream_out=stream_out,
                stream_err=stream_err,
                logger=logger,
            )
        else:
            self._state = state
            if indent and indent != self._state.indent_unit:
                self._state.indent_unit = indent

    @staticmethod
    def _resolve_debug(
        force_debug: Optional[bool], show_debug: Optional[bool], debug_env: Optional[str]
    ) -> bool:
        if force_debug is not None:
            return force_debug
        if show_debug is not None:
            return show_debug
        if not debug_env:
            return True
        env_value = os.environ.get(debug_env)
        if env_value is None:
            return True
        return env_value.lower() not in _FALSEY

    @property
    def show_debug(self) -> bool:
        return self._state.show_debug

    def child(self, context: str) -> "StructuredReporter":
        """Create a child reporter with an additional context tag."""
        return StructuredReporter(
            contexts=self._contexts + (context,),
            indent=self._state.indent_unit,
            stream_out=self._state.stream_out,
            stream_err=self._state.stream_err,
            flush=self._state.flush,
            logger=self._state.logger,
            state=self._state,
        )

    def _emit(self, level: str, message: str, *, contexts: Optional[Iterable[str]] = None) -> None:
        if level == "DEBUG" and not self._state.show_debug:
            return

        active_contexts = tuple(contexts) if contexts is not None else self._contexts
        prefix_parts = [f"[{level}]"] + [f"[{ctx}]" for ctx in active_contexts]
        prefix = "".join(prefix_parts)
        indent = self._state.indent_unit * self._state.indent
        formatted = f"{prefix} {indent}{message}" if message else prefix

        stream = self._state.stream_err if level in _WARNING_LEVELS else self._state.stream_out
        print(formatted, file=stream, flush=self._state.flush)

        if self._state.logger:
            log_method = getattr(self._state.logger, level.lower(), self._state.logger.info)
            log_method(formatted)

    def debug(self, message: str) -> None:
        self._emit("DEBUG", message)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warning(self, message: str) -> None:
        self._emit("WARNING", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def exception(self, message: str, error: BaseException) -> None:
        self.error(f"{message}: {error}")
        import traceback

        traceback.print_exc(file=self._state.stream_err)

    def item(self, message: str, *, level: str = "INFO") -> None:
        self._emit(level, f"- {message}")

    @contextmanager
    def section(
        self,
        title: str,
        *,
        level: str = "INFO",
        context: Optional[str] = None,
    ):
        reporter = self if context is None else self.child(context)
        current_contexts = reporter._contexts if context is not None else None
        should_emit = level != "DEBUG" or self._state.show_debug
        if should_emit:
            self._emit(level, title, contexts=current_contexts)
            self._state.indent += 1
        try:
            yield reporter
        finally:
            if should_emit:
                self._state.indent = max(0, self._state.indent - 1)

    def optional_debug_section(self, title: str, *, context: Optional[str] = None):
        if not self._state.show_debug:
            return nullcontext(self if context is None else self.child(context))
        return self.section(title, level="DEBUG", context=context)
