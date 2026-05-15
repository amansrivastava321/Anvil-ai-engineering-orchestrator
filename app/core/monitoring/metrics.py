"""Prometheus metrics for the orchestrator.

Exports required by orchestrator.py:
    agent_executions, agent_execution_duration, agent_tokens_used,
    successful_code_generations, MetricsTracker
"""

import time
from typing import Any, Dict

from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "agent_executions",
    "agent_execution_duration",
    "agent_tokens_used",
    "successful_code_generations",
    "active_executions",
    "model_requests",
    "model_latency",
    "model_errors",
    "active_connections",
    "graphify_context_size",
    "graphify_parsing_duration",
    "MetricsTracker",
    "setup_metrics",
]

# ── Counters ──────────────────────────────────────────────────────────────────

agent_executions: Counter = Counter(
    "agent_executions_total",
    "Total agent execution attempts",
    ["agent_type", "model_used", "status"],
)

agent_tokens_used: Counter = Counter(
    "agent_tokens_used_total",
    "Total tokens consumed by agents",
    ["agent_type", "model_used"],
)

successful_code_generations: Counter = Counter(
    "successful_code_generations_total",
    "Successful code generation tasks",
)

model_requests: Counter = Counter(
    "model_requests_total",
    "Total model API requests",
    ["model_name", "provider"],
)

model_errors: Counter = Counter(
    "model_errors_total",
    "Total model API errors",
    ["model_name", "error_type"],
)

# ── Histograms ────────────────────────────────────────────────────────────────

agent_execution_duration: Histogram = Histogram(
    "agent_execution_duration_seconds",
    "Agent execution duration",
    ["agent_type"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

model_latency: Histogram = Histogram(
    "model_latency_seconds",
    "Model API request latency",
    ["model_name", "provider"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

active_executions: Gauge = Gauge(
    "active_executions_current",
    "Currently active agent executions",
)

active_connections: Gauge = Gauge(
    "active_connections_current",
    "Currently active model API connections",
)

# ── Graphify context metrics ──────────────────────────────────────────────────

graphify_context_size: Histogram = Histogram(
    "graphify_context_size_tokens",
    "Assembled context size in approximate tokens",
    buckets=[128, 512, 1024, 2048, 4096, 8192, 16384, 32768],
)

graphify_parsing_duration: Histogram = Histogram(
    "graphify_parsing_duration_seconds",
    "Time to parse and assemble graphify context",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


def setup_metrics(app: object | None = None) -> None:
    """Instrument the FastAPI app with Prometheus metrics middleware."""
    if app is None:
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import]
        Instrumentator().instrument(app).expose(app)  # type: ignore[arg-type]
    except Exception:
        pass


class MetricsTracker:
    """Async context manager: tracks duration + active execution count."""

    def __init__(self, histogram: Histogram, labels: Dict[str, str]) -> None:
        self._histogram = histogram
        self._labels = labels
        self._start: float = 0.0

    async def __aenter__(self) -> "MetricsTracker":
        self._start = time.monotonic()
        active_executions.inc()
        return self

    async def __aexit__(self, *_: Any) -> None:
        duration = time.monotonic() - self._start
        self._histogram.labels(**self._labels).observe(duration)
        active_executions.dec()
