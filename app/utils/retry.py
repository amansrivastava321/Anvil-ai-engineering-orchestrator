"""Retry utilities with exponential backoff and circuit breaker."""

import asyncio
import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Type

import structlog

logger = structlog.get_logger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for retry policy decisions."""

    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behaviour."""

    max_attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0
    max_delay: float = 60.0
    retryable_categories: Tuple[ErrorCategory, ...] = field(
        default_factory=lambda: (
            ErrorCategory.TRANSIENT,
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.SERVER_ERROR,
            ErrorCategory.NETWORK,
        )
    )

__all__ = ["CircuitBreaker", "CircuitBreakerOpenError", "async_retry", "ErrorCategory", "RetryConfig"]


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open and calls are blocked."""

    def __init__(self, message: str = "Circuit breaker is open") -> None:
        super().__init__(message)


class CircuitBreaker:
    """
    Simple circuit breaker with three states: CLOSED → OPEN → HALF_OPEN → CLOSED.

    Transitions:
    - CLOSED → OPEN: failure_threshold consecutive failures
    - OPEN → HALF_OPEN: after recovery_timeout seconds
    - HALF_OPEN → CLOSED: success_threshold consecutive successes
    - HALF_OPEN → OPEN: any failure
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None

    def _check_half_open_transition(self) -> None:
        """Advance OPEN → HALF_OPEN if recovery_timeout has elapsed."""
        if self._state == self.OPEN:
            if self._last_failure_time is None:
                return
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed > self.recovery_timeout:
                self._state = self.HALF_OPEN
                self._success_count = 0

    @property
    def state(self) -> str:
        self._check_half_open_transition()
        return self._state

    def can_attempt(self) -> bool:
        self._check_half_open_transition()
        return self._state in (self.CLOSED, self.HALF_OPEN)

    def is_open(self) -> bool:
        self._check_half_open_transition()
        return self._state == self.OPEN

    def record_success(self) -> None:
        self._check_half_open_transition()
        if self._state == self.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = self.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed")
        elif self._state == self.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        self._check_half_open_transition()
        self._last_failure_time = time.monotonic()
        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            self._success_count = 0
            logger.warning("Circuit breaker reopened from half-open")
            return
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                "Circuit breaker opened",
                failures=self._failure_count,
                threshold=self.failure_threshold,
            )

    async def __aenter__(self) -> "CircuitBreaker":
        if not self.can_attempt():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open (state={self._state})"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: retry an async function with exponential backoff."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if max_attempts < 1:
                raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

            current_delay = delay
            last_exc: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "Retrying after failure",
                            func=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay_s=round(current_delay, 2),
                            error=str(exc),
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
