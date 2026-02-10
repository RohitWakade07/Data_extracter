"""
Structured Logging Framework
=============================

Provides enterprise-grade logging with:
- Structured logging (JSON format)
- Request correlation IDs
- Performance metrics
- Error tracking
"""

import logging
import logging.handlers
import json
import time
import os
from typing import Any, Dict, Optional
from functools import wraps
from datetime import datetime
import sys


class StructuredFormatter(logging.Formatter):
    """Format logs as structured JSON for better parsing"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present (using getattr for type safety)
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id is not None:
            log_data["correlation_id"] = correlation_id

        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        service = getattr(record, "service", None)
        if service is not None:
            log_data["service"] = service

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_structured: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        use_structured: Whether to use structured JSON format

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5  # 10MB per file
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_performance(func):
    """Decorator to log function execution time and parameters"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"{func.__name__} completed successfully",
                extra={"duration_ms": duration_ms},
            )
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"{func.__name__} failed: {str(e)}",
                extra={"duration_ms": duration_ms},
                exc_info=True,
            )
            raise

    return wrapper


def log_with_context(correlation_id: str, service: str):
    """Decorator to add context to all logs within a function"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)

            # Create context filter
            class ContextFilter(logging.Filter):
                def filter(self, record):
                    record.correlation_id = correlation_id
                    record.service = service
                    return True

            # Add filter temporarily
            context_filter = ContextFilter()
            logger.addFilter(context_filter)

            try:
                return func(*args, **kwargs)
            finally:
                logger.removeFilter(context_filter)

        return wrapper

    return decorator


class Logger:
    """Convenience wrapper for logger instances"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = setup_logger(name, level)
        self.name = name

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs, exc_info=exc_info)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)

    def performance(self, duration_ms: float, operation: str, **kwargs):
        """Log performance metric"""
        self.logger.info(
            f"Performance: {operation}",
            extra={"duration_ms": duration_ms, **kwargs},
        )
