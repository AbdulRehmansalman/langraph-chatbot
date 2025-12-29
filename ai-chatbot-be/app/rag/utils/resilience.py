"""
Resilience Patterns
===================
Circuit breaker and retry patterns for fault-tolerant RAG operations.

This module provides enterprise-grade resilience patterns:
- CircuitBreaker: Prevents cascading failures
- with_retry: Automatic retry with exponential backoff
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by stopping calls to failing services.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Calls are blocked, returns immediately
    - HALF-OPEN: Limited calls allowed to test recovery

    Usage:
        breaker = CircuitBreaker()
        if breaker.can_execute():
            try:
                result = await some_operation()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half-open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_requests: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            half_open_requests: Successful requests needed to close
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.failures = 0
        self.last_failure_time = 0.0
        self.state = self.STATE_CLOSED
        self.half_open_successes = 0

    def can_execute(self) -> bool:
        """Check if circuit allows execution."""
        if self.state == self.STATE_CLOSED:
            return True
        elif self.state == self.STATE_OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = self.STATE_HALF_OPEN
                self.half_open_successes = 0
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
        else:  # half-open
            return True

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == self.STATE_HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self.state = self.STATE_CLOSED
                self.failures = 0
                logger.info("Circuit breaker closed after successful recovery")
        elif self.state == self.STATE_CLOSED:
            self.failures = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = self.STATE_OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failures} failures"
            )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.failures = 0
        self.state = self.STATE_CLOSED
        self.half_open_successes = 0
        logger.info("Circuit breaker reset")

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == self.STATE_OPEN


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated async function with retry logic

    Usage:
        @with_retry(max_retries=3, delay=1.0)
        async def flaky_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}"
                        )
            raise last_exception

        return wrapper

    return decorator


def with_retry_sync(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Synchronous version of retry decorator.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated sync function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}"
                        )
            raise last_exception

        return wrapper

    return decorator
