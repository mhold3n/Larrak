"""Tests for logging functionality aligned with project logging policy."""

import logging

from campro.logging import get_logger


class TestLogging:
    """Test logging module functionality."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a logging.Logger instance with correct name."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_sets_correct_level(self):
        """Logger is set to INFO level by default."""
        logger = get_logger("test_module")
        assert logger.level == logging.INFO

    def test_get_logger_does_not_add_handlers(self):
        """Module-level get_logger must not add handlers (policy)."""
        # Capture handler count before
        logger = logging.getLogger("policy_test")
        before = len(logger.handlers)
        _ = get_logger("policy_test")
        after = len(logger.handlers)
        assert after == before

    def test_logger_reuse(self):
        """Same name returns the same logger instance."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        assert logger1 is logger2

    def test_different_loggers_for_different_names(self):
        """Different names return distinct logger instances."""
        logger1 = get_logger("test_module_1")
        logger2 = get_logger("test_module_2")
        assert logger1 is not logger2
        assert logger1.name != logger2.name
