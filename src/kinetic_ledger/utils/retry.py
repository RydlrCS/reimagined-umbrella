"""
Retry policies with exponential backoff and jitter.
"""
import random
from typing import Callable, Type
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from .errors import DependencyError, RpcError, GeminiError, CircleError, VectorDbError


logger = logging.getLogger(__name__)


def with_jitter(base: float, multiplier: float = 1.0) -> float:
    """Add jitter to wait time."""
    return base * multiplier * (0.5 + random.random())


# Standard retry policy for external API calls
def retry_on_dependency_error(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
) -> Callable:
    """
    Retry decorator for external dependency calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type((
            DependencyError,
            RpcError,
            GeminiError,
            CircleError,
            VectorDbError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Gemini-specific retry (more aggressive due to rate limits)
def retry_on_gemini_error(
    max_attempts: int = 5,
    min_wait: int = 2,
    max_wait: int = 30,
) -> Callable:
    """Retry decorator for Gemini API calls."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type(GeminiError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# RPC retry (blockchain calls)
def retry_on_rpc_error(
    max_attempts: int = 4,
    min_wait: int = 1,
    max_wait: int = 15,
) -> Callable:
    """Retry decorator for blockchain RPC calls."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RpcError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
