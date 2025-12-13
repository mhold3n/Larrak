from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TextIO

_DEFAULT_DEBUG_ENV = "CAMPRO_DEBUG"
_FALSEY = {"0", "false", "no", "off"}
_WARNING_LEVELS = {"WARNING", "ERROR", "CRITICAL"}


@dataclass
class _ReporterState:
    indent: int
    indent_unit: str
    flush: bool
    show_debug: bool
    console_min_level: int
    stream_out: TextIO
    stream_err: TextIO
    logger: logging.Logger | None


class StructuredReporter:
    """Structured console reporter with hierarchical context support."""

    def __init__(
        self,
        *,
        context: str | None = None,
        contexts: tuple[str, ...] | None = None,
        indent: str = "  ",
        force_debug: bool | None = None,
        show_debug: bool | None = None,
        debug_env: str | None = _DEFAULT_DEBUG_ENV,
        console_min_level: str | int = "INFO",
        stream_out: TextIO = sys.stdout,
        stream_err: TextIO = sys.stderr,
        flush: bool = True,
        logger: logging.Logger | None = None,
        state: _ReporterState | None = None,
    ) -> None:
        if contexts is not None:
            self._contexts: tuple[str, ...] = tuple(contexts)
        elif context:
            self._contexts = (context,)
        else:
            self._contexts = ()

        if state is None:
            resolved_debug = self._resolve_debug(force_debug, show_debug, debug_env)

            # Resolve console minimum level
            if isinstance(console_min_level, str):
                min_level_int = getattr(logging, console_min_level.upper(), logging.INFO)
            else:
                min_level_int = int(console_min_level)

            self._state = _ReporterState(
                indent=0,
                indent_unit=indent,
                flush=flush,
                show_debug=resolved_debug,
                console_min_level=min_level_int,
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
        force_debug: bool | None,
        show_debug: bool | None,
        debug_env: str | None,
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

    def child(self, context: str) -> StructuredReporter:
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

    def _emit(self, level: str, message: str, *, contexts: Iterable[str] | None = None) -> None:
        # Determine numeric level for this message
        msg_level_int = getattr(logging, level.upper(), logging.INFO)

        # 1. Emit to Logger (if configured)
        # We pass everything to the logger, letting the logger's own handlers decide filtering.
        # However, we respect show_debug for DEBUG messages if they are not meant to be seen at all.
        # Actually, if show_debug is False, we shouldn't generate DEBUG messages at all?
        # The original code returned early if level=="DEBUG" and not show_debug.
        # But now we might want DEBUG in file but not console.
        # So we should only return early if we DON'T want it anywhere.
        # Let's assume show_debug controls "generation" of debug messages.
        # If show_debug is True, we generate them.

        if level == "DEBUG" and not self._state.show_debug:
            return

        active_contexts = tuple(contexts) if contexts is not None else self._contexts
        prefix_parts = [f"[{level}]"] + [f"[{ctx}]" for ctx in active_contexts]
        prefix = "".join(prefix_parts)
        indent = self._state.indent_unit * self._state.indent
        formatted = f"{prefix} {indent}{message}" if message else prefix

        # 2. Emit to Console (if meets min level AND not disabled globally)
        if (
            msg_level_int >= self._state.console_min_level
            and msg_level_int > logging.root.manager.disable
        ):
            stream = self._state.stream_err if level in _WARNING_LEVELS else self._state.stream_out
            print(formatted, file=stream, flush=self._state.flush)

        # 3. Emit to Logger
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
        context: str | None = None,
    ) -> Iterator[StructuredReporter]:
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

    def optional_debug_section(
        self, title: str, *, context: str | None = None
    ) -> AbstractContextManager[StructuredReporter]:
        if not self._state.show_debug:
            return nullcontext(self if context is None else self.child(context))  # type: ignore[return-value]
        return self.section(title, level="DEBUG", context=context)
