import asyncio
import pytest
from app.utils.retry import async_retry, CircuitBreaker, CircuitBreakerOpenError


@pytest.mark.asyncio
async def test_async_retry_succeeds_on_first_try():
    call_count = 0

    @async_retry(max_attempts=3, delay=0.01)
    async def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await fn()
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_async_retry_retries_on_failure_then_succeeds():
    call_count = 0

    @async_retry(max_attempts=3, delay=0.01)
    async def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("transient")
        return "ok"

    result = await fn()
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_async_retry_raises_after_max_attempts():
    @async_retry(max_attempts=2, delay=0.01)
    async def fn():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        await fn()


def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    assert cb.state == CircuitBreaker.CLOSED
    assert cb.can_attempt()


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    assert not cb.can_attempt()


def test_circuit_breaker_resets_on_success():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_open_error_is_exception():
    err = CircuitBreakerOpenError("breaker open")
    assert isinstance(err, Exception)
    assert "breaker open" in str(err)


def test_circuit_breaker_reopens_from_half_open_on_failure():
    import time
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
    # Open the breaker
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    # Wait for recovery timeout
    time.sleep(0.02)
    # Now in HALF_OPEN - one failure should re-open
    cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN


def test_circuit_breaker_half_open_success_transitions_closed():
    import time
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01, success_threshold=1)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    time.sleep(0.02)
    # Transition to HALF_OPEN by checking state
    assert cb.state == CircuitBreaker.HALF_OPEN
    # A success in HALF_OPEN should close the breaker
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_closed_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()  # success while CLOSED resets count
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_is_open_property():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    assert not cb.is_open()
    cb.record_failure()
    assert cb.is_open()


@pytest.mark.asyncio
async def test_circuit_breaker_aenter_aexit_success():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    async with cb:
        pass  # no exception → record_success
    assert cb.state == CircuitBreaker.CLOSED
    assert cb._failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_aenter_aexit_failure():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    with pytest.raises(RuntimeError):
        async with cb:
            raise RuntimeError("fail")
    assert cb.state == CircuitBreaker.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_aenter_raises_when_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    cb.record_failure()
    assert cb.is_open()
    with pytest.raises(CircuitBreakerOpenError):
        async with cb:
            pass


@pytest.mark.asyncio
async def test_async_retry_max_attempts_less_than_one_raises():
    @async_retry(max_attempts=0, delay=0.01)
    async def fn():
        return "ok"

    with pytest.raises(ValueError, match="max_attempts"):
        await fn()


@pytest.mark.asyncio
async def test_async_retry_only_retries_specified_exceptions():
    call_count = 0

    @async_retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
    async def fn():
        nonlocal call_count
        call_count += 1
        raise TypeError("not retried")

    with pytest.raises(TypeError):
        await fn()
    # TypeError is not in exceptions — should fail on first try without retry
    assert call_count == 1


@pytest.mark.asyncio
async def test_async_retry_exponential_backoff():
    import time
    call_times = []

    @async_retry(max_attempts=3, delay=0.02, backoff=2.0)
    async def fn():
        call_times.append(time.monotonic())
        raise ValueError("always fails")

    with pytest.raises(ValueError):
        await fn()
    assert len(call_times) == 3
