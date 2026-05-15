import asyncio
import pytest
from unittest.mock import MagicMock
from app.core.monitoring.metrics import (
    MetricsTracker,
    agent_executions,
    active_executions,
    agent_execution_duration,
    agent_tokens_used,
    successful_code_generations,
)


@pytest.mark.asyncio
async def test_metrics_tracker_observes_duration():
    mock_histogram = MagicMock()
    mock_labeled = MagicMock()
    mock_histogram.labels.return_value = mock_labeled

    async with MetricsTracker(mock_histogram, {"agent_type": "code"}):
        await asyncio.sleep(0.01)

    mock_histogram.labels.assert_called_once_with(agent_type="code")
    mock_labeled.observe.assert_called_once()
    observed = mock_labeled.observe.call_args[0][0]
    assert observed >= 0.01


@pytest.mark.asyncio
async def test_metrics_tracker_increments_active_gauge():
    before = active_executions._value.get()

    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = MagicMock()

    async with MetricsTracker(mock_histogram, {}):
        during = active_executions._value.get()

    after = active_executions._value.get()
    assert during == before + 1
    assert after == before


def test_agent_executions_counter_exists():
    agent_executions.labels(agent_type="code", model_used="qwen", status="started")


def test_agent_tokens_used_counter_exists():
    agent_tokens_used.labels(agent_type="code", model_used="qwen")


def test_successful_code_generations_counter_exists():
    successful_code_generations.inc(0)


def test_agent_execution_duration_histogram_exists():
    agent_execution_duration.labels(agent_type="code")


def test_setup_metrics_returns_early_when_app_none():
    from app.core.monitoring.metrics import setup_metrics
    setup_metrics()       # default app=None → early return
    setup_metrics(None)   # explicit None → early return


def test_setup_metrics_with_app_ignores_missing_package():
    from app.core.monitoring.metrics import setup_metrics
    # Passing any object triggers the import attempt;
    # prometheus_fastapi_instrumentator may or may not be installed.
    # Either way the function must not raise.
    setup_metrics(object())
