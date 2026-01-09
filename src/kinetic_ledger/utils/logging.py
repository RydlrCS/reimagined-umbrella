"""
Structured JSON logging with correlation IDs for distributed tracing.
"""
import json
import logging
import os
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID from context
        correlation_id = correlation_id_var.get()
        if correlation_id:
            payload["correlation_id"] = correlation_id

        # Add extra fields
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = getattr(record, "correlation_id")
        
        if hasattr(record, "service"):
            payload["service"] = getattr(record, "service")

        # Add exception info if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(service_name: str = "kinetic-ledger") -> None:
    """Configure structured JSON logging."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    # Create JSON handler
    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Set service name as extra field
    logging.getLogger().addFilter(lambda record: setattr(record, "service", service_name) or True)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from context."""
    return correlation_id_var.get()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
