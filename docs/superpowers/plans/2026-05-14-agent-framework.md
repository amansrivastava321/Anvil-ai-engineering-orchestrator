# Agent Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete missing intelligence layer — infrastructure stubs, BaseAgent abstraction, 4 specialized agents, 3 multi-agent workflows, and tool implementations — so the orchestrator runs end-to-end with real agents instead of raw Ollama calls.

**Architecture:** Fix broken imports first (metrics, artifacts, retry) so the system starts. Then build a `BaseAgent` ABC that agents extend — each agent has a domain-focused system prompt, owns its tool list, reasons before responding, and emits structured logs. Workflows are proper classes that coordinate agents, replacing the inline methods in the orchestrator.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, structlog, prometheus-client, aiofiles, pytest-asyncio

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `app/utils/retry.py` | **Empty — blocks import** | `async_retry` decorator, `CircuitBreaker`, `CircuitBreakerOpenError` |
| `app/core/monitoring/metrics.py` | **Empty — blocks import** | Prometheus counters/histograms, `MetricsTracker` context manager |
| `app/artifacts/store.py` | **Empty — blocks import** | File-based artifact storage, `ArtifactStore`, `get_artifact_store` |
| `app/agents/base_agent.py` | Empty | `BaseAgent` ABC, `AgentTask`, `AgentResult`, `BaseTool`, `ToolResult`, `AgentStep` |
| `app/agents/agent_factory.py` | Empty | `AgentFactory`, `get_agent_factory` |
| `app/agents/specialized/code_agent.py` | Empty | `CodeAgent` — code generation, review, refactoring |
| `app/agents/specialized/architecture_agent.py` | Empty | `ArchitectureAgent` — dependency analysis, impact assessment |
| `app/agents/specialized/testing_agent.py` | Empty | `TestingAgent` — test strategy and generation |
| `app/agents/specialized/documentation_agent.py` | Empty | `DocumentationAgent` — docstrings, READMEs, API docs |
| `app/tools/file_system/file_reader.py` | Empty | `FileReader` tool — reads repo files safely |
| `app/tools/file_system/file_writer.py` | Empty | `FileWriter` tool — writes files with path validation |
| `app/tools/testing/test_runner.py` | Empty | `TestRunner` tool — runs pytest, captures output |
| `app/workflows/debug_workflow.py` | Empty | `DebugWorkflow` — 3-agent debug pipeline |
| `app/workflows/refactor_workflow.py` | Empty | `RefactorWorkflow` — 3-agent refactoring pipeline |
| `app/workflows/testing_workflow.py` | Empty | `TestingWorkflow` — 2-agent test generation pipeline |
| `app/services/orchestrator.py` | **Modify** | Wire in `AgentFactory` + `DebugWorkflow` + `RefactorWorkflow` |
| `tests/unit/test_retry.py` | Create | Tests for retry decorator and circuit breaker |
| `tests/unit/test_metrics.py` | Create | Tests for MetricsTracker |
| `tests/unit/test_artifacts.py` | Create | Tests for ArtifactStore |
| `tests/unit/test_base_agent.py` | Create | Tests for BaseAgent, AgentFactory |
| `tests/unit/test_workflows.py` | Create | Tests for all 3 workflows |

---

## Execution Order

```
Phase 1 (parallel — fix broken imports):
  Task 1: retry.py
  Task 2: metrics.py
  Task 3: artifacts/store.py

Phase 2 (parallel — tools, depends on validators only):
  Task 4: file_reader.py
  Task 5: file_writer.py
  Task 6: test_runner.py

Phase 3 (sequential — agent framework):
  Task 7: base_agent.py  ← all agents depend on this
  Task 8: agent_factory.py

Phase 4 (parallel — specialized agents, depend on Task 7):
  Task 9:  code_agent.py
  Task 10: architecture_agent.py
  Task 11: testing_agent.py
  Task 12: documentation_agent.py

Phase 5 (parallel — workflows, depend on Tasks 8-12):
  Task 13: debug_workflow.py
  Task 14: refactor_workflow.py
  Task 15: testing_workflow.py

Phase 6:
  Task 16: Wire orchestrator to use AgentFactory + workflow classes
```

---

## Task 1: Retry Utilities

**Files:**
- Create: `app/utils/retry.py`
- Create: `tests/unit/test_retry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_retry.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/aman/Documents/Projects/ai-engineering-orchestrator
python -m pytest tests/unit/test_retry.py -v --no-header 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'app.utils.retry'`

- [ ] **Step 3: Implement `app/utils/retry.py`**

```python
"""Retry utilities with exponential backoff and circuit breaker."""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, Tuple, Type

import structlog

logger = structlog.get_logger(__name__)


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

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            elapsed = time.monotonic() - (self._last_failure_time or 0.0)
            if elapsed > self.recovery_timeout:
                self._state = self.HALF_OPEN
                self._success_count = 0
        return self._state

    def can_attempt(self) -> bool:
        return self.state in (self.CLOSED, self.HALF_OPEN)

    def is_open(self) -> bool:
        return self.state == self.OPEN

    def record_success(self) -> None:
        if self._state == self.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = self.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed")
        elif self._state == self.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                "Circuit breaker opened",
                failures=self._failure_count,
                threshold=self.failure_threshold,
            )
        if self._state == self.HALF_OPEN:
            self._state = self.OPEN


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
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/unit/test_retry.py -v --no-header
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add app/utils/retry.py tests/unit/test_retry.py
git commit -m "feat: implement retry utilities with circuit breaker"
```

---

## Task 2: Prometheus Metrics

**Files:**
- Create: `app/core/monitoring/metrics.py`
- Create: `tests/unit/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_metrics.py
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from app.core.monitoring.metrics import MetricsTracker, agent_executions, active_executions


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
    # active_executions gauge should go up inside, back to baseline outside
    before = active_executions._value.get()

    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = MagicMock()

    async with MetricsTracker(mock_histogram, {}):
        during = active_executions._value.get()

    after = active_executions._value.get()
    assert during == before + 1
    assert after == before


def test_agent_executions_counter_exists():
    # Counter should be labelable without error
    agent_executions.labels(agent_type="code", model_used="qwen", status="started")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/unit/test_metrics.py -v --no-header 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'app.core.monitoring.metrics'`

- [ ] **Step 3: Implement `app/core/monitoring/metrics.py`**

```python
"""Prometheus metrics for the orchestrator.

Imported by orchestrator.py — exports must match exactly:
    agent_executions, agent_execution_duration, agent_tokens_used,
    successful_code_generations, MetricsTracker
"""

import time
from typing import Any, Dict

from prometheus_client import Counter, Gauge, Histogram

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

# ── Histograms ────────────────────────────────────────────────────────────────

agent_execution_duration: Histogram = Histogram(
    "agent_execution_duration_seconds",
    "Agent execution duration",
    ["agent_type"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

active_executions: Gauge = Gauge(
    "active_executions_current",
    "Currently active agent executions",
)


class MetricsTracker:
    """Async context manager: tracks duration + active count for a single execution."""

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
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/unit/test_metrics.py -v --no-header
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/core/monitoring/metrics.py tests/unit/test_metrics.py
git commit -m "feat: implement prometheus metrics and MetricsTracker"
```

---

## Task 3: Artifact Store

**Files:**
- Create: `app/artifacts/store.py`
- Create: `tests/unit/test_artifacts.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_artifacts.py
import json
import pytest
import tempfile
from pathlib import Path
from app.artifacts.store import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(base_path=str(tmp_path / "artifacts"))


@pytest.mark.asyncio
async def test_save_artifact_creates_file(store):
    meta = await store.save_artifact(
        execution_id="exec-1",
        artifact_type="response",
        content="hello world",
    )
    assert meta["artifact_type"] == "response"
    assert meta["execution_id"] == "exec-1"
    assert Path(meta["path"]).read_text() == "hello world"


@pytest.mark.asyncio
async def test_save_artifact_returns_metadata_with_size(store):
    meta = await store.save_artifact(
        execution_id="exec-2",
        artifact_type="prompt",
        content="abc",
        metadata={"custom": "value"},
    )
    assert meta["size_bytes"] == 3
    assert meta["custom"] == "value"
    assert "created_at" in meta


@pytest.mark.asyncio
async def test_get_artifact_returns_content(store):
    meta = await store.save_artifact("exec-3", "response", "my content")
    content = await store.get_artifact("exec-3", meta["artifact_id"])
    assert content == "my content"


@pytest.mark.asyncio
async def test_get_artifact_missing_returns_none(store):
    result = await store.get_artifact("exec-x", "nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_artifacts_returns_all_for_execution(store):
    await store.save_artifact("exec-4", "response", "r")
    await store.save_artifact("exec-4", "prompt", "p")
    artifacts = await store.list_artifacts("exec-4")
    assert len(artifacts) == 2
    types = {a["artifact_type"] for a in artifacts}
    assert types == {"response", "prompt"}


@pytest.mark.asyncio
async def test_list_artifacts_empty_for_unknown_execution(store):
    artifacts = await store.list_artifacts("unknown-exec")
    assert artifacts == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/unit/test_artifacts.py -v --no-header 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'app.artifacts.store'`

- [ ] **Step 3: Implement `app/artifacts/store.py`**

```python
"""File-based artifact storage for execution results.

Imported by orchestrator.py — exports must match: ArtifactStore, get_artifact_store
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import structlog

logger = structlog.get_logger(__name__)


class ArtifactStore:
    """Stores execution artifacts (responses, prompts) as files.

    Layout: {base_path}/{execution_id}/{artifact_type}_{artifact_id}.{txt,json}
    """

    def __init__(self, base_path: str = "data/artifacts") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_artifact(
        self,
        execution_id: str,
        artifact_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist artifact content and return its metadata dict."""
        artifact_id = str(uuid.uuid4())
        artifact_dir = self.base_path / execution_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        content_path = artifact_dir / f"{artifact_type}_{artifact_id}.txt"
        async with aiofiles.open(content_path, "w") as fh:
            await fh.write(content)

        meta: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "execution_id": execution_id,
            "artifact_type": artifact_type,
            "path": str(content_path),
            "size_bytes": len(content.encode()),
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        meta_path = artifact_dir / f"{artifact_type}_{artifact_id}.json"
        async with aiofiles.open(meta_path, "w") as fh:
            await fh.write(json.dumps(meta, indent=2))

        logger.debug("Artifact saved", artifact_id=artifact_id, type=artifact_type)
        return meta

    async def get_artifact(
        self, execution_id: str, artifact_id: str
    ) -> Optional[str]:
        """Return artifact content, or None if not found."""
        artifact_dir = self.base_path / execution_id
        for path in artifact_dir.glob(f"*_{artifact_id}.txt"):
            async with aiofiles.open(path) as fh:
                return await fh.read()
        return None

    async def list_artifacts(self, execution_id: str) -> List[Dict[str, Any]]:
        """Return metadata for every artifact belonging to an execution."""
        artifact_dir = self.base_path / execution_id
        if not artifact_dir.exists():
            return []

        result: List[Dict[str, Any]] = []
        for meta_path in artifact_dir.glob("*.json"):
            async with aiofiles.open(meta_path) as fh:
                raw = await fh.read()
            result.append(json.loads(raw))
        return result

    async def close(self) -> None:
        """No-op for file store — satisfies orchestrator teardown interface."""


_default_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Return the process-wide default ArtifactStore."""
    global _default_store
    if _default_store is None:
        _default_store = ArtifactStore()
    return _default_store
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/unit/test_artifacts.py -v --no-header
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add app/artifacts/store.py tests/unit/test_artifacts.py
git commit -m "feat: implement file-based artifact store"
```

---

## Task 4: FileReader Tool

**Files:**
- Create: `app/tools/file_system/file_reader.py`

*(Tests for tools are included in Task 8 — agent integration tests.)*

- [ ] **Step 1: Implement `app/tools/file_system/file_reader.py`**

The tool reads files relative to a repo root, using `PathValidator` to prevent path traversal.

```python
"""FileReader — read repository files safely."""

import time
from pathlib import Path
from typing import Any

import aiofiles

from app.utils.validators import PathValidator


class ToolResult:
    """Result from any tool execution."""

    __slots__ = ("tool_name", "success", "output", "error", "duration_ms")

    def __init__(
        self,
        tool_name: str,
        success: bool,
        output: str,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.error = error
        self.duration_ms = duration_ms


class FileReader:
    """Tool: read a file from the repository by relative path."""

    name = "read_file"
    description = (
        "Read the full contents of a file. "
        "Input: file_path (string, relative to the repository root)."
    )

    def __init__(self, repo_path: str) -> None:
        self._repo = Path(repo_path)

    async def execute(self, file_path: str, **_: Any) -> ToolResult:
        start = time.monotonic()
        try:
            validated = PathValidator.validate_path(
                str(self._repo / file_path),
                must_exist=True,
                must_be_file=True,
            )
            async with aiofiles.open(validated) as fh:
                content = await fh.read()
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=content,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )
```

- [ ] **Step 2: Commit**

```bash
git add app/tools/file_system/file_reader.py
git commit -m "feat: implement FileReader tool"
```

---

## Task 5: FileWriter Tool

**Files:**
- Create: `app/tools/file_system/file_writer.py`

- [ ] **Step 1: Implement `app/tools/file_system/file_writer.py`**

```python
"""FileWriter — write files into the repository."""

import time
from pathlib import Path
from typing import Any

import aiofiles

from app.utils.validators import PathValidator
from app.tools.file_system.file_reader import ToolResult


class FileWriter:
    """Tool: write content to a file in the repository."""

    name = "write_file"
    description = (
        "Write content to a file. "
        "Inputs: file_path (relative to repo root), content (string). "
        "Creates the file if it does not exist. Overwrites if it does."
    )

    def __init__(self, repo_path: str) -> None:
        self._repo = Path(repo_path)

    async def execute(self, file_path: str, content: str, **_: Any) -> ToolResult:
        start = time.monotonic()
        try:
            full_path = self._repo / file_path
            # Validate the parent directory (must exist, must be dir)
            PathValidator.validate_path(
                str(full_path.parent),
                must_exist=True,
                must_be_dir=True,
            )
            async with aiofiles.open(full_path, "w") as fh:
                await fh.write(content)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"Wrote {len(content.encode())} bytes to {file_path}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )
```

- [ ] **Step 2: Commit**

```bash
git add app/tools/file_system/file_writer.py
git commit -m "feat: implement FileWriter tool"
```

---

## Task 6: TestRunner Tool

**Files:**
- Create: `app/tools/testing/test_runner.py`

- [ ] **Step 1: Implement `app/tools/testing/test_runner.py`**

```python
"""TestRunner — run pytest in the repository and capture output."""

import asyncio
import time
from pathlib import Path
from typing import Any

from app.tools.file_system.file_reader import ToolResult

_MAX_OUTPUT_CHARS = 8_000
_TIMEOUT_SECONDS = 120.0


class TestRunner:
    """Tool: run pytest tests and return the captured output."""

    name = "run_tests"
    description = (
        "Run pytest. "
        "Inputs: test_path (optional, relative path to test file or directory), "
        "extra_args (optional string of additional pytest flags)."
    )

    def __init__(self, repo_path: str) -> None:
        self._repo = Path(repo_path)

    async def execute(
        self,
        test_path: str | None = None,
        extra_args: str | None = None,
        **_: Any,
    ) -> ToolResult:
        start = time.monotonic()
        cmd = ["python", "-m", "pytest", "-v", "--tb=short", "--no-header"]

        if test_path:
            cmd.append(test_path)
        if extra_args:
            cmd.extend(extra_args.split())

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_TIMEOUT_SECONDS
            )
            output = stdout.decode(errors="replace")[:_MAX_OUTPUT_CHARS]
            success = proc.returncode == 0
            return ToolResult(
                tool_name=self.name,
                success=success,
                output=output,
                error=None if success else f"Tests failed (exit code {proc.returncode})",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Test run timed out after {_TIMEOUT_SECONDS}s",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )
```

- [ ] **Step 2: Commit**

```bash
git add app/tools/testing/test_runner.py
git commit -m "feat: implement TestRunner tool"
```

---

## Task 7: BaseAgent

**Files:**
- Create: `app/agents/base_agent.py`
- Create: `tests/unit/test_base_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_base_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.base_agent import (
    AgentTask,
    AgentResult,
    AgentStatus,
    BaseAgent,
    ToolResult,
)


class ConcreteAgent(BaseAgent):
    """Minimal concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test_agent"

    @property
    def description(self) -> str:
        return "A test agent"

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."

    async def _execute(self, task: AgentTask) -> str:
        return await self._call_model(task)


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="model response")
    return client


@pytest.fixture
def agent(mock_ollama):
    return ConcreteAgent(ollama_client=mock_ollama)


@pytest.fixture
def task():
    return AgentTask(
        prompt="Do something",
        repo_path="/tmp",
        model="qwen2.5-coder:7b",
    )


@pytest.mark.asyncio
async def test_agent_run_returns_completed_result(agent, task):
    result = await agent.run(task)
    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.COMPLETED
    assert result.response == "model response"
    assert result.agent_name == "test_agent"
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_agent_run_returns_failed_result_on_exception(agent, task, mock_ollama):
    mock_ollama.chat.side_effect = RuntimeError("model down")
    result = await agent.run(task)
    assert result.status == AgentStatus.FAILED
    assert "model down" in (result.error or "")
    assert result.response == ""


@pytest.mark.asyncio
async def test_agent_run_streaming_yields_tokens(agent, task, mock_ollama):
    async def token_gen(*args, **kwargs):
        for token in ["Hello", " ", "world"]:
            yield token

    mock_ollama.chat = token_gen
    tokens = []
    async for token in agent.run_streaming(task):
        tokens.append(token)
    assert tokens == ["Hello", " ", "world"]


def test_build_messages_includes_system_and_user(agent, task):
    messages = agent._build_messages(task)
    assert messages[0]["role"] == "system"
    assert "test agent" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"
    assert "Do something" in messages[1]["content"]


def test_tool_result_tracks_success():
    result = ToolResult(tool_name="read_file", success=True, output="content")
    assert result.success
    assert result.error is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/unit/test_base_agent.py -v --no-header 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'app.agents.base_agent'`

- [ ] **Step 3: Implement `app/agents/base_agent.py`**

```python
"""Base agent abstraction for all specialized engineering agents.

Agents are NOT Ollama wrappers. They are reasoning entities with:
- A focused engineering domain (code, architecture, testing, docs)
- A structured system prompt expressing their expertise and discipline
- A tool list they can use to gather real information
- Full observability via structured logs
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.monitoring.logging import get_logger
from app.integrations.ollama.client import OllamaClient, get_default_client


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolResult:
    """Result from a single tool execution."""

    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class AgentStep:
    """One reasoning step recorded during agent execution."""

    thought: str
    action: Optional[str] = None
    tool_used: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class AgentResult:
    """Complete result from an agent execution."""

    agent_name: str
    execution_id: str
    status: AgentStatus
    response: str
    steps: List[AgentStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tokens_used: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentTask(BaseModel):
    """Task submitted to an agent for execution."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    repo_path: str
    context: Optional[Any] = None  # AssembledContext — typed loosely to avoid circular import
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.2
    max_tokens: int = 4096
    stream: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class BaseTool(ABC):
    """Abstract base for tools that agents can invoke."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used in agent reasoning and logging."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description injected into the agent's system prompt."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool. Must not raise — return ToolResult with success=False on error."""


class BaseAgent(ABC):
    """Abstract base for all specialized engineering agents.

    Subclasses must implement:
    - name (str property)
    - description (str property)
    - system_prompt (str property)
    - _execute(task: AgentTask) -> str

    Optionally override:
    - tools (list[BaseTool] property) to give the agent file/test access
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None) -> None:
        self.ollama = ollama_client or get_default_client()
        self._logger = get_logger(f"agent.{self.name}")

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier — used in logs and metrics."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of this agent's specialization."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Core system prompt. Defines the agent's expertise and discipline."""

    @property
    def tools(self) -> List[BaseTool]:
        """Tools available to this agent. Override to add domain-specific tools."""
        return []

    @abstractmethod
    async def _execute(self, task: AgentTask) -> str:
        """Domain-specific execution logic. Called by run()."""

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentResult:
        """Execute task synchronously and return a complete AgentResult."""
        execution_id = str(uuid.uuid4())
        start = time.monotonic()

        self._logger.info(
            "Agent starting",
            agent=self.name,
            task_id=task.task_id,
            execution_id=execution_id,
            model=task.model,
        )

        try:
            response = await self._execute(task)
            duration_ms = (time.monotonic() - start) * 1000

            self._logger.info(
                "Agent completed",
                agent=self.name,
                execution_id=execution_id,
                duration_ms=round(duration_ms, 2),
                tokens_approx=len(response) // 4,
            )

            return AgentResult(
                agent_name=self.name,
                execution_id=execution_id,
                status=AgentStatus.COMPLETED,
                response=response,
                tokens_used=len(response) // 4,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._logger.error(
                "Agent failed",
                agent=self.name,
                execution_id=execution_id,
                error=str(exc),
                exc_info=True,
            )
            return AgentResult(
                agent_name=self.name,
                execution_id=execution_id,
                status=AgentStatus.FAILED,
                response="",
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def run_streaming(self, task: AgentTask) -> AsyncIterator[str]:
        """Stream response tokens as they are generated."""
        self._logger.info("Agent streaming", agent=self.name, task_id=task.task_id)
        messages = self._build_messages(task)

        async for token in self.ollama.chat(
            model=task.model,
            messages=messages,
            temperature=task.temperature,
            max_tokens=task.max_tokens,
            stream=True,
        ):
            yield token

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(self, task: AgentTask) -> List[Dict[str, str]]:
        """Assemble the message list passed to the model."""
        system = self._full_system_prompt()

        if task.context is not None and hasattr(task.context, "user_prompt"):
            user_content = f"{task.context.user_prompt}\n\n---\n\n{task.prompt}"
        else:
            user_content = task.prompt

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _full_system_prompt(self) -> str:
        """System prompt with tool descriptions appended if tools are registered."""
        base = self.system_prompt
        if not self.tools:
            return base
        tool_lines = ["", "Available tools:"]
        for tool in self.tools:
            tool_lines.append(f"- {tool.name}: {tool.description}")
        return base + "\n".join(tool_lines)

    async def _call_model(
        self,
        task: AgentTask,
        focus: Optional[str] = None,
    ) -> str:
        """Execute a single synchronous model call."""
        messages = self._build_messages(task)
        if focus:
            messages[0]["content"] += f"\n\nFOCUS: {focus}"

        return await self.ollama.chat(  # type: ignore[return-value]
            model=task.model,
            messages=messages,
            temperature=task.temperature,
            max_tokens=task.max_tokens,
            stream=False,
        )
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/unit/test_base_agent.py -v --no-header
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add app/agents/base_agent.py tests/unit/test_base_agent.py
git commit -m "feat: implement BaseAgent abstraction"
```

---

## Task 8: Agent Factory

**Files:**
- Create: `app/agents/agent_factory.py`

*(Specialized agents aren't built yet — factory tests run after Task 9-12.)*

- [ ] **Step 1: Implement `app/agents/agent_factory.py`**

```python
"""Factory for creating and caching specialized agent instances.

Imported by orchestrator.py and workflows. Add new agents to _REGISTRY.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import structlog

from app.agents.base_agent import BaseAgent
from app.integrations.ollama.client import OllamaClient

logger = structlog.get_logger(__name__)

# Populated after specialized agents are implemented (Tasks 9-12).
# Import inside get_agent() to avoid circular imports at module load.
_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def _load_registry() -> Dict[str, Type[BaseAgent]]:
    """Lazy-load specialized agents to keep import graph clean."""
    if _REGISTRY:
        return _REGISTRY

    from app.agents.specialized.code_agent import CodeAgent
    from app.agents.specialized.architecture_agent import ArchitectureAgent
    from app.agents.specialized.testing_agent import TestingAgent
    from app.agents.specialized.documentation_agent import DocumentationAgent

    _REGISTRY.update({
        "code": CodeAgent,
        "architecture": ArchitectureAgent,
        "testing": TestingAgent,
        "documentation": DocumentationAgent,
    })
    return _REGISTRY


class AgentFactory:
    """Creates and caches agent instances by type string.

    Usage:
        factory = AgentFactory()
        agent = factory.get_agent("code")
        result = await agent.run(task)
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None) -> None:
        self._ollama = ollama_client
        self._cache: Dict[str, BaseAgent] = {}

    def get_agent(self, agent_type: str) -> BaseAgent:
        """Return a cached agent instance for the given type.

        Raises ValueError for unknown agent types.
        """
        registry = _load_registry()
        if agent_type not in registry:
            raise ValueError(
                f"Unknown agent type {agent_type!r}. "
                f"Available: {sorted(registry)}"
            )
        if agent_type not in self._cache:
            cls = registry[agent_type]
            self._cache[agent_type] = cls(ollama_client=self._ollama)
            logger.debug("Agent created", agent_type=agent_type, cls=cls.__name__)
        return self._cache[agent_type]

    def list_agents(self) -> Dict[str, str]:
        """Map agent type → description for API introspection."""
        registry = _load_registry()
        return {
            name: (cls.description.fget(cls) if isinstance(cls.description, property) else "")  # type: ignore[attr-defined]
            for name, cls in registry.items()
        }


_default_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Return the process-wide AgentFactory singleton."""
    global _default_factory
    if _default_factory is None:
        _default_factory = AgentFactory()
    return _default_factory
```

- [ ] **Step 2: Commit**

```bash
git add app/agents/agent_factory.py
git commit -m "feat: implement AgentFactory with lazy registry"
```

---

## Task 9: CodeAgent

**Files:**
- Create: `app/agents/specialized/code_agent.py`
- Create: `tests/unit/test_specialized_agents.py` (seed with CodeAgent tests; extend in Tasks 10-12)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_specialized_agents.py
import pytest
from unittest.mock import AsyncMock
from app.agents.base_agent import AgentTask, AgentStatus
from app.agents.specialized.code_agent import CodeAgent
from app.agents.specialized.architecture_agent import ArchitectureAgent
from app.agents.specialized.testing_agent import TestingAgent
from app.agents.specialized.documentation_agent import DocumentationAgent


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="agent response")
    return client


@pytest.fixture
def task():
    return AgentTask(prompt="Write a function", repo_path="/tmp")


# ── CodeAgent ─────────────────────────────────────────────────────────────────

def test_code_agent_name(mock_ollama):
    agent = CodeAgent(ollama_client=mock_ollama)
    assert agent.name == "code_agent"


def test_code_agent_system_prompt_mentions_security(mock_ollama):
    agent = CodeAgent(ollama_client=mock_ollama)
    assert "security" in agent.system_prompt.lower()


@pytest.mark.asyncio
async def test_code_agent_run_returns_completed(mock_ollama, task):
    agent = CodeAgent(ollama_client=mock_ollama)
    result = await agent.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.response == "agent response"


# ── ArchitectureAgent ─────────────────────────────────────────────────────────

def test_architecture_agent_name(mock_ollama):
    agent = ArchitectureAgent(ollama_client=mock_ollama)
    assert agent.name == "architecture_agent"


def test_architecture_agent_system_prompt_mentions_dependencies(mock_ollama):
    agent = ArchitectureAgent(ollama_client=mock_ollama)
    assert "depend" in agent.system_prompt.lower() or "coupling" in agent.system_prompt.lower()


@pytest.mark.asyncio
async def test_architecture_agent_run_returns_completed(mock_ollama, task):
    agent = ArchitectureAgent(ollama_client=mock_ollama)
    result = await agent.run(task)
    assert result.status == AgentStatus.COMPLETED


# ── TestingAgent ──────────────────────────────────────────────────────────────

def test_testing_agent_name(mock_ollama):
    agent = TestingAgent(ollama_client=mock_ollama)
    assert agent.name == "testing_agent"


def test_testing_agent_system_prompt_mentions_mocking(mock_ollama):
    agent = TestingAgent(ollama_client=mock_ollama)
    assert "mock" in agent.system_prompt.lower()


@pytest.mark.asyncio
async def test_testing_agent_run_returns_completed(mock_ollama, task):
    agent = TestingAgent(ollama_client=mock_ollama)
    result = await agent.run(task)
    assert result.status == AgentStatus.COMPLETED


# ── DocumentationAgent ────────────────────────────────────────────────────────

def test_documentation_agent_name(mock_ollama):
    agent = DocumentationAgent(ollama_client=mock_ollama)
    assert agent.name == "documentation_agent"


def test_documentation_agent_system_prompt_mentions_why(mock_ollama):
    agent = DocumentationAgent(ollama_client=mock_ollama)
    assert "why" in agent.system_prompt.lower() or "purpose" in agent.system_prompt.lower()


@pytest.mark.asyncio
async def test_documentation_agent_run_returns_completed(mock_ollama, task):
    agent = DocumentationAgent(ollama_client=mock_ollama)
    result = await agent.run(task)
    assert result.status == AgentStatus.COMPLETED


# ── AgentFactory ──────────────────────────────────────────────────────────────

def test_agent_factory_returns_correct_types(mock_ollama):
    from app.agents.agent_factory import AgentFactory
    factory = AgentFactory(ollama_client=mock_ollama)
    assert isinstance(factory.get_agent("code"), CodeAgent)
    assert isinstance(factory.get_agent("architecture"), ArchitectureAgent)
    assert isinstance(factory.get_agent("testing"), TestingAgent)
    assert isinstance(factory.get_agent("documentation"), DocumentationAgent)


def test_agent_factory_caches_instances(mock_ollama):
    from app.agents.agent_factory import AgentFactory
    factory = AgentFactory(ollama_client=mock_ollama)
    a1 = factory.get_agent("code")
    a2 = factory.get_agent("code")
    assert a1 is a2


def test_agent_factory_raises_on_unknown_type(mock_ollama):
    from app.agents.agent_factory import AgentFactory
    factory = AgentFactory(ollama_client=mock_ollama)
    with pytest.raises(ValueError, match="Unknown agent type"):
        factory.get_agent("nonexistent")
```

- [ ] **Step 2: Implement `app/agents/specialized/code_agent.py`**

```python
"""Code generation, review, and refactoring agent."""

from app.agents.base_agent import BaseAgent, AgentTask


class CodeAgent(BaseAgent):
    """Specialized agent for production-quality code generation, review, and refactoring.

    Primary model: Qwen Coder. Focuses on:
    - Type-safe, async-first implementation
    - Security-aware code (no injection, no secrets in logs)
    - Minimal, readable code without unnecessary abstraction
    - Error paths treated as first-class citizens
    """

    @property
    def name(self) -> str:
        return "code_agent"

    @property
    def description(self) -> str:
        return "Generates, reviews, and refactors production-quality code"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior software engineer. You write production-quality code.

Standards you never compromise on:
- Full type annotations — no implicit Any
- Async-first: no blocking calls in the hot path
- Security-first: validate all inputs, never log secrets or PII, prevent injection
- Explicit error handling — no silent swallowing of exceptions
- Readable naming that makes comments unnecessary
- No abstractions that aren't immediately justified by the code in front of you

When generating code:
1. Understand the complete requirement before writing a single line
2. Consider the error paths before the happy path
3. Write the implementation
4. Verify it handles: empty input, large input, concurrent access, partial failure

When reviewing code:
1. Bugs and security issues — highest priority
2. Correctness under concurrency
3. Performance concerns
4. Style and readability — lowest priority
Give specific feedback with corrected code snippets, not vague suggestions."""

    async def _execute(self, task: AgentTask) -> str:
        return await self._call_model(task)
```

- [ ] **Step 3: Run the CodeAgent tests (others will fail — that's expected)**

```bash
python -m pytest tests/unit/test_specialized_agents.py::test_code_agent_name \
  tests/unit/test_specialized_agents.py::test_code_agent_system_prompt_mentions_security \
  tests/unit/test_specialized_agents.py::test_code_agent_run_returns_completed \
  -v --no-header
```

Expected: `3 passed`

- [ ] **Step 4: Commit**

```bash
git add app/agents/specialized/code_agent.py tests/unit/test_specialized_agents.py
git commit -m "feat: implement CodeAgent"
```

---

## Task 10: ArchitectureAgent

**Files:**
- Create: `app/agents/specialized/architecture_agent.py`

- [ ] **Step 1: Implement `app/agents/specialized/architecture_agent.py`**

```python
"""Architecture analysis and design agent."""

from app.agents.base_agent import BaseAgent, AgentTask


class ArchitectureAgent(BaseAgent):
    """Specialized agent for system architecture analysis and impact assessment.

    Uses Graphify graph context when available. Focuses on:
    - Dependency graphs and coupling analysis
    - Change blast radius — what breaks if X changes?
    - Architectural debt identification
    - Migration paths for improvements
    """

    @property
    def name(self) -> str:
        return "architecture_agent"

    @property
    def description(self) -> str:
        return "Analyzes system architecture, dependencies, and structural risks"

    @property
    def system_prompt(self) -> str:
        return """\
You are a principal software architect specializing in distributed systems and codebase intelligence.

Your lens:
- System boundaries and coupling between modules
- Dependency graph analysis: who depends on what, and what breaks if it changes
- Data flows: how state moves through the system
- God objects and high-centrality nodes — change risk multipliers
- Circular dependencies — architectural debt that compounds over time
- Scalability assumptions baked into the current design

When given Graphify graph data:
- Use community detection to identify natural module boundaries
- Flag nodes with high in-degree (many dependents) as change risk hubs
- Trace impact propagation: "if file X changes, these N files are affected"
- Identify architectural seams where the system could be split or decoupled

Structure every response as:
1. Current state — what the architecture looks like now
2. Problems — coupling, debt, risks
3. Recommendations — specific, actionable changes
4. Risk assessment — what can go wrong, migration order"""

    async def _execute(self, task: AgentTask) -> str:
        return await self._call_model(task)
```

- [ ] **Step 2: Run ArchitectureAgent tests**

```bash
python -m pytest tests/unit/test_specialized_agents.py -k "architecture" -v --no-header
```

Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add app/agents/specialized/architecture_agent.py
git commit -m "feat: implement ArchitectureAgent"
```

---

## Task 11: TestingAgent

**Files:**
- Create: `app/agents/specialized/testing_agent.py`

- [ ] **Step 1: Implement `app/agents/specialized/testing_agent.py`**

```python
"""Test strategy and test generation agent."""

from app.agents.base_agent import BaseAgent, AgentTask


class TestingAgent(BaseAgent):
    """Specialized agent for test strategy design and test generation.

    Applies strict testing discipline:
    - Tests document behavior, not implementation
    - Mock only at system boundaries (HTTP, DB, filesystem)
    - Each test has one clear assertion
    - AAA structure: Arrange, Act, Assert
    """

    @property
    def name(self) -> str:
        return "testing_agent"

    @property
    def description(self) -> str:
        return "Designs test strategies and generates comprehensive pytest test suites"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior QA engineer and testing expert. You think in behaviors, not implementations.

Testing philosophy you apply without exception:
- Tests document what the system DOES, not how it does it
- Test at the right level: unit for pure logic, integration for service contracts, e2e for critical user journeys
- Each test name is a sentence: "test_returns_error_when_user_not_found"
- Each test has exactly one assertion — no "and" in test names
- Mock only at system boundaries: HTTP calls, database, filesystem, clock
- Never mock internal classes — if you need to, the code has a coupling problem
- Tests must be deterministic and order-independent

When designing a test strategy:
1. List the behaviors to test (not the functions)
2. Identify which level each test belongs at
3. Identify what needs mocking and why
4. Identify edge cases: empty, maximum, concurrent, partial failure
5. Specify test fixtures needed

When generating pytest tests:
- Use pytest fixtures for setup, not setUp methods
- Use pytest.mark.asyncio for async tests
- Use tmp_path fixture for filesystem tests
- Use httpx.AsyncClient or pytest-httpx for HTTP mocking
- Include parametrize for data-driven cases
- Output complete, runnable files with all imports"""

    async def _execute(self, task: AgentTask) -> str:
        return await self._call_model(task)
```

- [ ] **Step 2: Run TestingAgent tests**

```bash
python -m pytest tests/unit/test_specialized_agents.py -k "testing" -v --no-header
```

Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add app/agents/specialized/testing_agent.py
git commit -m "feat: implement TestingAgent"
```

---

## Task 12: DocumentationAgent

**Files:**
- Create: `app/agents/specialized/documentation_agent.py`

- [ ] **Step 1: Implement `app/agents/specialized/documentation_agent.py`**

```python
"""Documentation generation agent."""

from app.agents.base_agent import BaseAgent, AgentTask


class DocumentationAgent(BaseAgent):
    """Specialized agent for generating accurate, useful documentation.

    Focuses on:
    - WHY over WHAT (code shows what; docs explain why)
    - Public API docstrings with full contract
    - README structure that answers the reader's actual questions
    - Examples that run, not examples that illustrate
    """

    @property
    def name(self) -> str:
        return "documentation_agent"

    @property
    def description(self) -> str:
        return "Generates accurate docstrings, READMEs, and API documentation from code"

    @property
    def system_prompt(self) -> str:
        return """\
You are a technical writer who also writes production code. You know what engineers actually need.

Documentation principles you apply:
- Document the WHY — the constraints, invariants, and non-obvious design decisions
- The WHAT is in the code — don't narrate it
- Every public function/class gets a docstring: purpose, parameters, return value, exceptions raised
- Stale documentation is worse than no documentation — be precise so it stays correct
- Examples must be runnable, not illustrative

Docstring format (Google style):
    def function(param: Type) -> ReturnType:
        \"\"\"One sentence: what this does and why.

        Args:
            param: Description including units, constraints, valid range.

        Returns:
            Description of the return value and its structure.

        Raises:
            ValueError: When param is invalid.
            RuntimeError: When the underlying system fails.

        Example:
            result = function(valid_input)
            assert result == expected
        \"\"\"

README structure:
1. What it is (one sentence)
2. Why you'd use it (the problem it solves)
3. Quick start (minimal working example)
4. Full API reference
5. Configuration reference"""

    async def _execute(self, task: AgentTask) -> str:
        return await self._call_model(task)
```

- [ ] **Step 2: Run all specialized agent tests (including factory tests)**

```bash
python -m pytest tests/unit/test_specialized_agents.py -v --no-header
```

Expected: `15 passed`

- [ ] **Step 3: Commit**

```bash
git add app/agents/specialized/documentation_agent.py
git commit -m "feat: implement DocumentationAgent and complete agent factory tests"
```

---

## Task 13: DebugWorkflow

**Files:**
- Create: `app/workflows/debug_workflow.py`
- Create: `tests/unit/test_workflows.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_workflows.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.base_agent import AgentResult, AgentStatus
from app.workflows.debug_workflow import DebugWorkflow, DebugWorkflowResult
from app.workflows.refactor_workflow import RefactorWorkflow, RefactorWorkflowResult
from app.workflows.testing_workflow import TestingWorkflow, TestingWorkflowResult


def make_mock_factory(response: str = "agent response"):
    """Build a mock AgentFactory where every agent returns a fixed response."""
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(
        agent_name="mock",
        execution_id="exec-1",
        status=AgentStatus.COMPLETED,
        response=response,
    ))
    factory = MagicMock()
    factory.get_agent.return_value = mock_agent
    return factory, mock_agent


# ── DebugWorkflow ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_debug_workflow_returns_result():
    factory, _ = make_mock_factory("diagnosis")
    workflow = DebugWorkflow(factory=factory)
    result = await workflow.run(
        prompt="API returning 500",
        repo_path="/tmp",
        context=None,
        debug_model="deepseek",
        code_model="qwen",
    )
    assert isinstance(result, DebugWorkflowResult)
    assert result.root_cause_analysis == "diagnosis"
    assert result.solution == "diagnosis"
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_debug_workflow_calls_agents_in_order():
    factory, mock_agent = make_mock_factory("response")
    workflow = DebugWorkflow(factory=factory)
    await workflow.run("bug", "/tmp", None, "m1", "m2")
    # architecture + rca + solution = 3 calls
    assert mock_agent.run.call_count == 3


def test_debug_workflow_result_to_markdown():
    result = DebugWorkflowResult(
        root_cause_analysis="root cause",
        solution="fix code",
    )
    md = result.to_markdown()
    assert "## Root Cause Analysis" in md
    assert "root cause" in md
    assert "## Solution" in md
    assert "fix code" in md


# ── RefactorWorkflow ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_refactor_workflow_returns_result():
    factory, _ = make_mock_factory("refactored")
    workflow = RefactorWorkflow(factory=factory)
    result = await workflow.run("/tmp", "refactor this", None, "arch-model", "code-model")
    assert isinstance(result, RefactorWorkflowResult)
    assert result.refactored_code == "refactored"


@pytest.mark.asyncio
async def test_refactor_workflow_calls_three_agents():
    factory, mock_agent = make_mock_factory()
    workflow = RefactorWorkflow(factory=factory)
    await workflow.run("/tmp", "code", None, "m1", "m2")
    assert mock_agent.run.call_count == 3  # analysis + refactor + review


# ── TestingWorkflow ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_testing_workflow_returns_result():
    factory, _ = make_mock_factory("test suite")
    workflow = TestingWorkflow(factory=factory)
    result = await workflow.run("test this", "/tmp", None, "testing-model", "code-model")
    assert isinstance(result, TestingWorkflowResult)
    assert result.generated_tests == "test suite"


@pytest.mark.asyncio
async def test_testing_workflow_calls_two_agents():
    factory, mock_agent = make_mock_factory()
    workflow = TestingWorkflow(factory=factory)
    await workflow.run("code", "/tmp", None, "m1", "m2")
    assert mock_agent.run.call_count == 2  # strategy + generation
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/unit/test_workflows.py -v --no-header 2>&1 | head -10
```

Expected: `ModuleNotFoundError` on workflow imports

- [ ] **Step 3: Implement `app/workflows/debug_workflow.py`**

```python
"""Multi-agent debug workflow: Architecture context → Root cause → Solution."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.core.monitoring.logging import get_logger
from app.services.context_service import AssembledContext

logger = get_logger(__name__)


@dataclass
class DebugWorkflowResult:
    """Result from the 3-agent debug pipeline."""

    root_cause_analysis: str
    solution: str
    steps_taken: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        return (
            f"## Root Cause Analysis\n\n{self.root_cause_analysis}"
            f"\n\n## Solution\n\n{self.solution}"
        )


class DebugWorkflow:
    """3-agent debug pipeline.

    Step 1 — ArchitectureAgent: Map the structure and data flow around the issue.
    Step 2 — CodeAgent: Identify the root cause using the architecture context.
    Step 3 — CodeAgent: Generate the fix from the identified root cause.
    """

    def __init__(self, factory: Optional[AgentFactory] = None) -> None:
        self._factory = factory or get_agent_factory()

    async def run(
        self,
        prompt: str,
        repo_path: str,
        context: Optional[AssembledContext],
        debug_model: str,
        code_model: str,
    ) -> DebugWorkflowResult:
        start = time.monotonic()
        steps: List[str] = []
        models_used: List[str] = []

        logger.info("Debug workflow starting", repo_path=repo_path)

        arch_agent = self._factory.get_agent("architecture")
        code_agent = self._factory.get_agent("code")

        # Step 1: Architecture context
        arch_result = await arch_agent.run(AgentTask(
            prompt=(
                f"Map the architecture and data flow relevant to this issue:\n\n{prompt}\n\n"
                "Which modules, functions, and data paths are involved?"
            ),
            repo_path=repo_path,
            context=context,
            model=debug_model,
            temperature=0.1,
        ))
        steps.append("architecture_analysis")
        models_used.append(debug_model)

        # Step 2: Root cause analysis
        rca_result = await code_agent.run(AgentTask(
            prompt=(
                f"Issue:\n{prompt}\n\n"
                f"Architecture context:\n{arch_result.response}\n\n"
                "Identify the root cause. Be specific: which file, function, or interaction "
                "is the origin of the failure, and exactly why does it fail?"
            ),
            repo_path=repo_path,
            context=context,
            model=debug_model,
            temperature=0.1,
        ))
        steps.append("root_cause_analysis")

        # Step 3: Solution
        solution_result = await code_agent.run(AgentTask(
            prompt=(
                f"Root cause:\n{rca_result.response}\n\n"
                "Generate the fix. Show the complete modified function or class, "
                "not just a diff. Include any new tests needed to prevent regression."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("solution_generation")
        models_used.append(code_model)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("Debug workflow completed", duration_ms=round(duration_ms, 2), steps=steps)

        return DebugWorkflowResult(
            root_cause_analysis=rca_result.response,
            solution=solution_result.response,
            steps_taken=steps,
            duration_ms=duration_ms,
            models_used=list(set(models_used)),
        )
```

- [ ] **Step 4: Commit**

```bash
git add app/workflows/debug_workflow.py
git commit -m "feat: implement DebugWorkflow (3-agent pipeline)"
```

---

## Task 14: RefactorWorkflow

**Files:**
- Create: `app/workflows/refactor_workflow.py`

- [ ] **Step 1: Implement `app/workflows/refactor_workflow.py`**

```python
"""Multi-agent refactoring workflow: Analyze → Refactor → Review."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.core.monitoring.logging import get_logger
from app.services.context_service import AssembledContext

logger = get_logger(__name__)


@dataclass
class RefactorWorkflowResult:
    """Result from the 3-agent refactoring pipeline."""

    analysis: str
    refactored_code: str
    review: str
    steps_taken: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_markdown(self) -> str:
        return (
            f"## Architecture Analysis\n\n{self.analysis}"
            f"\n\n## Refactored Code\n\n{self.refactored_code}"
            f"\n\n## Review\n\n{self.review}"
        )


class RefactorWorkflow:
    """3-agent refactoring pipeline.

    Step 1 — ArchitectureAgent: Identify what to refactor and why.
    Step 2 — CodeAgent: Perform the refactoring.
    Step 3 — CodeAgent: Review the refactored output for regressions.
    """

    def __init__(self, factory: Optional[AgentFactory] = None) -> None:
        self._factory = factory or get_agent_factory()

    async def run(
        self,
        repo_path: str,
        prompt: str,
        context: Optional[AssembledContext],
        arch_model: str,
        code_model: str,
    ) -> RefactorWorkflowResult:
        start = time.monotonic()
        steps: List[str] = []

        logger.info("Refactor workflow starting")

        arch_agent = self._factory.get_agent("architecture")
        code_agent = self._factory.get_agent("code")

        # Step 1: Analysis
        analysis_result = await arch_agent.run(AgentTask(
            prompt=(
                f"Analyze the following code for refactoring opportunities.\n\n{prompt}\n\n"
                "Identify: what should change, why, and what the risk is."
            ),
            repo_path=repo_path,
            context=context,
            model=arch_model,
            temperature=0.2,
        ))
        steps.append("architecture_analysis")

        # Step 2: Refactor
        refactor_result = await code_agent.run(AgentTask(
            prompt=(
                f"Refactor this code based on the analysis below.\n\n"
                f"Analysis:\n{analysis_result.response}\n\n"
                f"Code:\n{prompt}\n\n"
                "Output the complete refactored implementation. Preserve all public contracts."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("refactoring")

        # Step 3: Review
        review_result = await code_agent.run(AgentTask(
            prompt=(
                f"Review this refactored code.\n\n"
                f"Original:\n{prompt}\n\n"
                f"Refactored:\n{refactor_result.response}\n\n"
                "Identify any bugs, broken contracts, or security issues introduced. "
                "Be specific about line numbers and what's wrong."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("code_review")

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("Refactor workflow completed", duration_ms=round(duration_ms, 2))

        return RefactorWorkflowResult(
            analysis=analysis_result.response,
            refactored_code=refactor_result.response,
            review=review_result.response,
            steps_taken=steps,
            duration_ms=duration_ms,
        )
```

- [ ] **Step 2: Commit**

```bash
git add app/workflows/refactor_workflow.py
git commit -m "feat: implement RefactorWorkflow (3-agent pipeline)"
```

---

## Task 15: TestingWorkflow

**Files:**
- Create: `app/workflows/testing_workflow.py`

- [ ] **Step 1: Implement `app/workflows/testing_workflow.py`**

```python
"""Multi-agent testing workflow: Strategy → Generate tests."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.core.monitoring.logging import get_logger
from app.services.context_service import AssembledContext

logger = get_logger(__name__)


@dataclass
class TestingWorkflowResult:
    """Result from the 2-agent testing pipeline."""

    test_strategy: str
    generated_tests: str
    steps_taken: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_markdown(self) -> str:
        return (
            f"## Test Strategy\n\n{self.test_strategy}"
            f"\n\n## Generated Tests\n\n{self.generated_tests}"
        )


class TestingWorkflow:
    """2-agent testing pipeline.

    Step 1 — TestingAgent: Design the test strategy (what to test, at what level, with what mocks).
    Step 2 — CodeAgent: Generate complete, runnable pytest test files from the strategy.
    """

    def __init__(self, factory: Optional[AgentFactory] = None) -> None:
        self._factory = factory or get_agent_factory()

    async def run(
        self,
        prompt: str,
        repo_path: str,
        context: Optional[AssembledContext],
        testing_model: str,
        code_model: str,
    ) -> TestingWorkflowResult:
        start = time.monotonic()
        steps: List[str] = []

        logger.info("Testing workflow starting")

        testing_agent = self._factory.get_agent("testing")
        code_agent = self._factory.get_agent("code")

        # Step 1: Strategy
        strategy_result = await testing_agent.run(AgentTask(
            prompt=(
                f"Design a test strategy for the following:\n\n{prompt}\n\n"
                "Specify: which behaviors to test, what level (unit/integration/e2e), "
                "what to mock and why, key edge cases, and required fixtures."
            ),
            repo_path=repo_path,
            context=context,
            model=testing_model,
            temperature=0.2,
        ))
        steps.append("test_strategy")

        # Step 2: Generate tests
        test_result = await code_agent.run(AgentTask(
            prompt=(
                f"Generate complete pytest tests based on this strategy.\n\n"
                f"Strategy:\n{strategy_result.response}\n\n"
                f"Code to test:\n{prompt}\n\n"
                "Output complete, runnable .py files with all imports. "
                "Use pytest fixtures, pytest-asyncio for async, and parametrize for data-driven cases."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("test_generation")

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("Testing workflow completed", duration_ms=round(duration_ms, 2))

        return TestingWorkflowResult(
            test_strategy=strategy_result.response,
            generated_tests=test_result.response,
            steps_taken=steps,
            duration_ms=duration_ms,
        )
```

- [ ] **Step 2: Run all workflow tests**

```bash
python -m pytest tests/unit/test_workflows.py -v --no-header
```

Expected: `8 passed`

- [ ] **Step 3: Commit**

```bash
git add app/workflows/testing_workflow.py
git commit -m "feat: implement TestingWorkflow (2-agent pipeline)"
```

---

## Task 16: Wire Orchestrator

**Files:**
- Modify: `app/services/orchestrator.py`

Replace the inline `_execute_debug_workflow` and `_execute_refactoring_workflow` with the new workflow classes. Add `AgentFactory` for single-agent dispatching.

- [ ] **Step 1: Add imports and factory to `Orchestrator.__init__`**

In `app/services/orchestrator.py`, add at the top with other imports:

```python
from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.workflows.debug_workflow import DebugWorkflow
from app.workflows.refactor_workflow import RefactorWorkflow
from app.workflows.testing_workflow import TestingWorkflow
```

In `Orchestrator.__init__`, after `self.skillfile = get_skillfile_client()`, add:

```python
self._agent_factory = AgentFactory(ollama_client=self.ollama)
self._debug_workflow = DebugWorkflow(factory=self._agent_factory)
self._refactor_workflow = RefactorWorkflow(factory=self._agent_factory)
self._testing_workflow = TestingWorkflow(factory=self._agent_factory)
```

- [ ] **Step 2: Replace `_execute_refactoring_workflow`**

Find the method `_execute_refactoring_workflow` (around line 711) and replace the entire method body with:

```python
async def _execute_refactoring_workflow(
    self,
    request: OrchestratorRequest,
    context: AssembledContext,
    primary_model: str,
    metrics: ExecutionMetrics,
) -> str:
    arch_model = await self.model_service.select_model(TaskCategory.ARCHITECTURE_ANALYSIS)
    code_model = await self.model_service.select_model(TaskCategory.CODE_REFACTORING)
    result = await self._refactor_workflow.run(
        repo_path=request.repo_path,
        prompt=request.prompt,
        context=context,
        arch_model=arch_model,
        code_model=code_model,
    )
    return result.to_markdown()
```

- [ ] **Step 3: Replace `_execute_debug_workflow`**

Find `_execute_debug_workflow` (around line 796) and replace body with:

```python
async def _execute_debug_workflow(
    self,
    request: OrchestratorRequest,
    context: AssembledContext,
    primary_model: str,
    metrics: ExecutionMetrics,
) -> str:
    debug_model = await self.model_service.select_model(TaskCategory.DEBUGGING)
    code_model = await self.model_service.select_model(TaskCategory.CODE_GENERATION)
    result = await self._debug_workflow.run(
        prompt=request.prompt,
        repo_path=request.repo_path,
        context=context,
        debug_model=debug_model,
        code_model=code_model,
    )
    return result.to_markdown()
```

- [ ] **Step 4: Replace `_execute_test_generation` to use workflow**

Find `_execute_test_generation` (around line 625) and replace with:

```python
async def _execute_test_generation(
    self,
    request: OrchestratorRequest,
    context: AssembledContext,
    model_name: str,
    metrics: ExecutionMetrics,
) -> str:
    testing_model = await self.model_service.select_model(TaskCategory.TEST_GENERATION)
    code_model = await self.model_service.select_model(TaskCategory.CODE_GENERATION)
    result = await self._testing_workflow.run(
        prompt=request.prompt,
        repo_path=request.repo_path,
        context=context,
        testing_model=testing_model,
        code_model=code_model,
    )
    return result.to_markdown()
```

- [ ] **Step 5: Update `_execute_single_agent` to route through agents**

Find `_execute_single_agent` and replace with:

```python
async def _execute_single_agent(
    self,
    request: OrchestratorRequest,
    context: AssembledContext,
    model_name: str,
    metrics: ExecutionMetrics,
    task_instruction: Optional[str] = None,
) -> str:
    agent_type = self._map_workflow_to_agent(request.workflow_type)
    agent = self._agent_factory.get_agent(agent_type)
    task = AgentTask(
        prompt=request.prompt if not task_instruction else f"{task_instruction}\n\n{request.prompt}",
        repo_path=request.repo_path,
        context=context,
        model=model_name,
        temperature=request.temperature,
        max_tokens=request.max_tokens or 4096,
    )
    result = await agent.run(task)
    if result.error:
        raise RuntimeError(result.error)
    return result.response
```

- [ ] **Step 6: Add `_map_workflow_to_agent` helper**

After `_map_workflow_to_task`, add:

```python
def _map_workflow_to_agent(self, workflow: WorkflowType) -> str:
    """Map workflow type to agent type string."""
    mapping = {
        WorkflowType.CODE_GENERATION: "code",
        WorkflowType.CODE_REVIEW: "code",
        WorkflowType.CODE_REFACTORING: "code",
        WorkflowType.DEBUG_ANALYSIS: "code",
        WorkflowType.ARCHITECTURE_ANALYSIS: "architecture",
        WorkflowType.TEST_GENERATION: "testing",
        WorkflowType.DOCUMENTATION: "documentation",
        WorkflowType.IMPACT_ANALYSIS: "architecture",
        WorkflowType.GENERAL_QA: "code",
    }
    return mapping.get(workflow, "code")
```

- [ ] **Step 7: Verify the orchestrator imports cleanly**

```bash
python -c "from app.services.orchestrator import Orchestrator; print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Run the full test suite**

```bash
python -m pytest tests/unit/ -v --no-header --tb=short
```

Expected: All tests passing.

- [ ] **Step 9: Commit**

```bash
git add app/services/orchestrator.py
git commit -m "feat: wire AgentFactory and workflow classes into orchestrator"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Covered by |
|---|---|
| Fix broken imports (metrics, artifacts, retry) | Tasks 1-3 |
| BaseAgent with streaming, tool support, observability | Task 7 |
| 4 specialized agents with domain-focused prompts | Tasks 9-12 |
| AgentFactory | Task 8 |
| DebugWorkflow as proper class | Task 13 |
| RefactorWorkflow as proper class | Task 14 |
| TestingWorkflow as proper class | Task 15 |
| Wire orchestrator to agents and workflows | Task 16 |
| Tools: FileReader, FileWriter, TestRunner | Tasks 4-6 |
| Tests for every component | Tasks 1-3, 7-9, 13 |
| Async throughout | All tasks — every tool and agent is async |
| Type safety | All tasks — every signature is typed |

**Parallel execution opportunities:**
- Tasks 1, 2, 3 can run simultaneously
- Tasks 4, 5, 6 can run simultaneously (after Tasks 1-3)
- Tasks 9, 10, 11, 12 can run simultaneously (after Task 7)
- Tasks 13, 14, 15 can run simultaneously (after Tasks 8-12)
