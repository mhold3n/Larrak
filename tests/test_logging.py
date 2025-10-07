"""Tests for logging functionality."""

import pytest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

from campro.logging import get_logger


class TestLogging:
    """Test logging module functionality."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logging.Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_sets_correct_level(self):
        """Test that logger is set to INFO level."""
        logger = get_logger("test_module")
        assert logger.level == logging.INFO
    
    def test_get_logger_has_handlers(self):
        """Test that logger has appropriate handlers."""
        logger = get_logger("test_module")
        assert len(logger.handlers) > 0
        
        # Check for console handler
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) > 0
    
    def test_get_logger_file_handler_when_logs_dir_exists(self):
        """Test that file handler is added when logs directory exists."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        try:
            logger = get_logger("test_module")
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) > 0
        finally:
            # Clean up
            if logs_dir.exists():
                logs_dir.rmdir()
    
    def test_get_logger_no_file_handler_when_logs_dir_missing(self):
        """Test that no file handler is added when logs directory doesn't exist."""
        # Ensure logs directory doesn't exist
        logs_dir = Path("logs")
        if logs_dir.exists():
            logs_dir.rmdir()
        
        logger = get_logger("test_module")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0
    
    def test_logger_reuse(self):
        """Test that the same logger instance is returned for the same name."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        assert logger1 is logger2
    
    def test_different_loggers_for_different_names(self):
        """Test that different logger instances are returned for different names."""
        logger1 = get_logger("test_module_1")
        logger2 = get_logger("test_module_2")
        assert logger1 is not logger2
        assert logger1.name != logger2.name





