"""Tests for app.services.orchestrator — Orchestrator core execution engine."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.orchestrator import (
    Orchestrator,
    OrchestratorRequest,
    OrchestratorResponse,
    ExecutionStatus,
    WorkflowType,
    ExecutionMode,
)


def _make_mock_context():
    ctx = MagicMock()
    ctx.total_tokens = 1000
    ctx.warnings = []
    ctx.system_prompt = "System prompt"
    ctx.user_prompt = "User prompt"
    ctx.graphify_available = False
    ctx.files_included = []
    ctx.skills_injected = 0
    ctx.skill_names = []
    ctx.budget_used = {}
    return ctx


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="AI response text")
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_model_service():
    svc = AsyncMock()
    svc.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
    svc.health_check = AsyncMock(return_value={"total_models": 1, "available_models": 1})
    return svc


@pytest.fixture
def mock_context_service():
    svc = AsyncMock()
    svc.assemble_context = AsyncMock(return_value=_make_mock_context())
    svc.get_stats = MagicMock(return_value={"total_assembled": 0})
    svc.clear_cache = MagicMock()
    svc.close = AsyncMock()
    return svc


@pytest.fixture
def mock_artifact_store():
    store = AsyncMock()
    store.save_run = AsyncMock(return_value={"run_id": "test-run-1"})
    store.save_artifact = AsyncMock(return_value={"artifact_id": "art-1", "type": "response"})
    store.close = AsyncMock()
    return store


@pytest.fixture
def orchestrator(mock_ollama, mock_model_service, mock_context_service, mock_artifact_store, tmp_path):
    with patch("app.services.orchestrator.get_default_client", return_value=mock_ollama), \
         patch("app.services.orchestrator.get_model_service", return_value=mock_model_service), \
         patch("app.services.orchestrator.get_context_service", return_value=mock_context_service), \
         patch("app.services.orchestrator.get_artifact_store", return_value=mock_artifact_store), \
         patch("app.services.orchestrator.get_default_parser") as mock_gp, \
         patch("app.services.orchestrator.get_skillfile_client") as mock_sf, \
         patch("app.services.orchestrator.AgentFactory") as mock_af, \
         patch("app.services.orchestrator.DebugWorkflow"), \
         patch("app.services.orchestrator.RefactorWorkflow"), \
         patch("app.services.orchestrator.TestingWorkflow"):
        mock_gp.return_value = MagicMock()
        mock_sf.return_value = MagicMock(is_installed=False)
        mock_af.return_value = MagicMock()
        orch = Orchestrator(
            ollama_client=mock_ollama,
            model_service=mock_model_service,
            context_service=mock_context_service,
            artifact_store=mock_artifact_store,
        )
        return orch


# ── get_active_executions ─────────────────────────────────────────────────────

def test_get_active_executions_empty(orchestrator):
    result = orchestrator.get_active_executions()
    assert result == []


def test_get_active_executions_with_entry(orchestrator):
    from app.services.orchestrator import ExecutionMetrics
    eid = "test-exec-1"
    orchestrator._active_executions[eid] = ExecutionMetrics(
        execution_id=eid,
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="qwen2.5-coder:7b",
        start_time=datetime.utcnow(),
    )
    result = orchestrator.get_active_executions()
    assert len(result) == 1
    assert result[0]["execution_id"] == eid


# ── get_execution_stats ────────────────────────────────────────────────────────

def test_get_execution_stats_empty(orchestrator):
    stats = orchestrator.get_execution_stats()
    assert stats == {"total": 0}


def test_get_execution_stats_with_history(orchestrator):
    from app.services.orchestrator import ExecutionMetrics
    m = ExecutionMetrics(
        execution_id="exec-1",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="qwen2.5-coder:7b",
        start_time=datetime.utcnow(),
        status=ExecutionStatus.COMPLETED,
        duration_ms=1500.0,
    )
    orchestrator._execution_history.append(m)
    stats = orchestrator.get_execution_stats()
    assert stats["total"] == 1
    assert stats["completed"] == 1
    assert stats["success_rate"] == 100.0


def test_get_execution_stats_with_failures(orchestrator):
    from app.services.orchestrator import ExecutionMetrics
    for status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.FAILED]:
        m = ExecutionMetrics(
            execution_id=f"exec-{status.value}",
            workflow_type=WorkflowType.GENERAL_QA,
            model_used="m",
            start_time=datetime.utcnow(),
            status=status,
            duration_ms=100.0,
        )
        orchestrator._execution_history.append(m)
    stats = orchestrator.get_execution_stats()
    assert stats["total"] == 3
    assert stats["completed"] == 1
    assert stats["failed"] == 2


# ── health_check ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check_returns_healthy(orchestrator, mock_ollama, mock_model_service, mock_context_service):
    mock_ollama.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_model_service.health_check = AsyncMock(return_value={"total_models": 2})
    mock_context_service.get_stats = MagicMock(return_value={"total_assembled": 0})
    result = await orchestrator.health_check()
    assert result["status"] == "healthy"
    assert "active_executions" in result


# ── ensure_fresh_intelligence ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ensure_fresh_intelligence_returns_status(orchestrator, tmp_path):
    with patch.object(orchestrator, "_auto_update_graphify", new=AsyncMock(
        return_value={"updated": False, "version": None}
    )):
        result = await orchestrator.ensure_fresh_intelligence(str(tmp_path))
    assert "graphify_updated" in result
    assert "skills_updated" in result


@pytest.mark.asyncio
async def test_ensure_fresh_intelligence_handles_error(orchestrator, tmp_path):
    with patch.object(orchestrator, "_auto_update_graphify", new=AsyncMock(
        side_effect=Exception("graphify error")
    )):
        result = await orchestrator.ensure_fresh_intelligence(str(tmp_path))
    assert result["graphify_updated"] is False


# ── execute ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_general_qa(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    from app.utils.validators import PathValidator
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    (tmp_path / ".git").mkdir()  # validate_repo_path requires .git directory
    PathValidator.set_allowed_base_dirs([tmp_path])
    try:
        mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
        mock_model_service.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
        mock_ollama.chat = AsyncMock(return_value="Here is the analysis...")
        with patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
            return_value={"graphify_updated": False, "skills_updated": False}
        )):
            req = OrchestratorRequest(
                repo_path=str(tmp_path),
                prompt="What is the architecture of this codebase?",
                workflow_type=WorkflowType.GENERAL_QA,
            )
            result = await orchestrator.execute(req)
        assert isinstance(result, OrchestratorResponse)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.response == "Here is the analysis..."
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


@pytest.mark.asyncio
async def test_execute_invalid_repo_returns_failed(orchestrator):
    with patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
        return_value={"graphify_updated": False, "skills_updated": False}
    )):
        req = MagicMock()
        req.repo_path = "/nonexistent/12345"
        req.prompt = "test"
        req.workflow_type = WorkflowType.GENERAL_QA
        req.mode = ExecutionMode.SYNC
        req.preferred_model = None
        req.include_files = None
        req.max_tokens = None
        req.temperature = 0.2
        req.conversation_id = None
        req.metadata = {}
        req.context_mode = "balanced"
        result = await orchestrator.execute(req)
        assert isinstance(result, OrchestratorResponse)
        assert result.status == ExecutionStatus.FAILED


# ── close ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_calls_cleanup(orchestrator, mock_ollama, mock_context_service, mock_artifact_store):
    mock_ollama.close = AsyncMock()
    mock_context_service.close = AsyncMock()
    mock_artifact_store.close = AsyncMock()
    await orchestrator.close()
    mock_ollama.close.assert_called_once()


# ── OrchestratorRequest validation ────────────────────────────────────────────

def test_orchestrator_request_invalid_repo_raises(tmp_path):
    from app.utils.validators import PathValidator
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        from pydantic import ValidationError
        with pytest.raises((ValidationError, Exception)):
            OrchestratorRequest(
                repo_path="/nonexistent/path/xyz",
                prompt="test",
            )
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


def test_orchestrator_response_defaults():
    resp = OrchestratorResponse(
        execution_id="test-1",
        status=ExecutionStatus.COMPLETED,
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="qwen2.5-coder:7b",
    )
    assert resp.response is None
    assert resp.tokens_used == 0
    assert resp.duration_ms == 0.0


# ── WorkflowType enumeration ──────────────────────────────────────────────────

def test_workflow_types_are_strings():
    assert isinstance(WorkflowType.GENERAL_QA.value, str)
    assert isinstance(WorkflowType.CODE_GENERATION.value, str)
    assert isinstance(WorkflowType.DEBUG_ANALYSIS.value, str)


# ── ExecutionStatus ────────────────────────────────────────────────────────────

def test_execution_status_values():
    assert ExecutionStatus.COMPLETED.value == "completed"
    assert ExecutionStatus.FAILED.value == "failed"
    assert ExecutionStatus.RUNNING.value == "running"


# ── execute with various workflow types ────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_code_generation(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    from app.utils.validators import PathValidator
    from app.services.orchestrator import WorkflowType
    (tmp_path / ".git").mkdir()
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    PathValidator.set_allowed_base_dirs([tmp_path])
    try:
        mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
        mock_model_service.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
        mock_ollama.chat = AsyncMock(return_value="Generated code here...")
        with patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
            return_value={"graphify_updated": False, "skills_updated": False}
        )):
            req = OrchestratorRequest(
                repo_path=str(tmp_path),
                prompt="Generate a REST API endpoint",
                workflow_type=WorkflowType.CODE_GENERATION,
            )
            result = await orchestrator.execute(req)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.workflow_type == WorkflowType.CODE_GENERATION
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


@pytest.mark.asyncio
async def test_execute_with_preferred_model(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    (tmp_path / ".git").mkdir()
    from app.utils.validators import PathValidator
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    PathValidator.set_allowed_base_dirs([tmp_path])
    try:
        mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
        mock_ollama.chat = AsyncMock(return_value="Using preferred model...")
        with patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
            return_value={"graphify_updated": False, "skills_updated": False}
        )):
            req = OrchestratorRequest(
                repo_path=str(tmp_path),
                prompt="Explain the code",
                workflow_type=WorkflowType.GENERAL_QA,
                preferred_model="llama3:8b",
            )
            result = await orchestrator.execute(req)
        # model_service.select_model should NOT be called when preferred_model is set
        mock_model_service.select_model.assert_not_called()
        assert result.model_used == "llama3:8b"
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


@pytest.mark.asyncio
async def test_execute_with_intelligence_refresh(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    (tmp_path / ".git").mkdir()
    from app.utils.validators import PathValidator
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    PathValidator.set_allowed_base_dirs([tmp_path])
    try:
        mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
        mock_model_service.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
        mock_ollama.chat = AsyncMock(return_value="Response...")
        with patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
            return_value={"graphify_updated": True, "skills_updated": True}
        )) as mock_refresh:
            req = OrchestratorRequest(
                repo_path=str(tmp_path),
                prompt="Analyze this",
                workflow_type=WorkflowType.GENERAL_QA,
            )
            result = await orchestrator.execute(req)
        mock_refresh.assert_called_once()
        assert result.status == ExecutionStatus.COMPLETED
        # Graphify and skills update warnings should be in response
        assert any("Graphify" in w for w in result.warnings)
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


@pytest.mark.asyncio
async def test_execute_ollama_error_returns_failed(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    (tmp_path / ".git").mkdir()
    from app.utils.validators import PathValidator
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    PathValidator.set_allowed_base_dirs([tmp_path])
    try:
        mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
        mock_model_service.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
        mock_ollama.chat = AsyncMock(side_effect=Exception("Connection refused"))
        with patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
            return_value={"graphify_updated": False, "skills_updated": False}
        )):
            req = OrchestratorRequest(
                repo_path=str(tmp_path),
                prompt="Analyze this",
                workflow_type=WorkflowType.GENERAL_QA,
            )
            result = await orchestrator.execute(req)
        assert result.status == ExecutionStatus.FAILED
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


# ── ExecutionMetrics ───────────────────────────────────────────────────────────

def test_execution_metrics_complete():
    from app.services.orchestrator import ExecutionMetrics
    m = ExecutionMetrics(
        execution_id="test-1",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="qwen2.5-coder:7b",
        start_time=datetime.utcnow(),
    )
    m.complete(tokens=500)
    assert m.status == ExecutionStatus.COMPLETED
    assert m.response_tokens == 500
    assert m.duration_ms >= 0


def test_execution_metrics_fail():
    from app.services.orchestrator import ExecutionMetrics
    m = ExecutionMetrics(
        execution_id="test-2",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="m",
        start_time=datetime.utcnow(),
    )
    m.fail("Something went wrong")
    assert m.status == ExecutionStatus.FAILED
    assert m.error == "Something went wrong"


def test_execution_metrics_fields():
    from app.services.orchestrator import ExecutionMetrics
    m = ExecutionMetrics(
        execution_id="test-3",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="m",
        start_time=datetime.utcnow(),
    )
    assert m.execution_id == "test-3"
    assert m.workflow_type == WorkflowType.GENERAL_QA
    assert m.status == ExecutionStatus.PENDING


# ── _estimate_tokens ───────────────────────────────────────────────────────────

def test_estimate_tokens(orchestrator):
    tokens = orchestrator._estimate_tokens("Hello " * 100)
    assert isinstance(tokens, int)
    assert tokens > 0


# ── _map_workflow_to_task ──────────────────────────────────────────────────────

def test_map_workflow_to_task_covers_all_types(orchestrator):
    from app.services.orchestrator import WorkflowType
    for wf in WorkflowType:
        task = orchestrator._map_workflow_to_task(wf)
        assert task is not None


# ── _error_response ────────────────────────────────────────────────────────────

def test_error_response_returns_failed(orchestrator):
    from app.services.orchestrator import ExecutionMetrics
    metrics = ExecutionMetrics(
        execution_id="err-1",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="qwen2.5-coder:7b",
        start_time=datetime.utcnow(),
    )
    resp = orchestrator._error_response(
        "err-1", WorkflowType.GENERAL_QA, metrics, "some error", ["w1"]
    )
    assert resp.status == ExecutionStatus.FAILED
    assert resp.error == "some error"
    assert "w1" in resp.warnings


# ── _execute_with_fallback ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_with_fallback_uses_fallback_model(orchestrator, mock_context_service, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
    mock_ollama.chat = AsyncMock(return_value="Fallback response")
    metrics = ExecutionMetrics(
        execution_id="fb-1",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="primary-model",
        start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.workflow_type = WorkflowType.GENERAL_QA
    req.prompt = "test"
    req.temperature = 0.2
    req.max_tokens = None
    req.metadata = {}
    resp = await orchestrator._execute_with_fallback(req, "fb-1", metrics)
    assert resp.status == ExecutionStatus.FALLBACK_USED
    assert resp.response == "Fallback response"


@pytest.mark.asyncio
async def test_execute_with_fallback_errors_return_error_response(orchestrator, mock_context_service, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_context_service.assemble_context = AsyncMock(side_effect=Exception("context failed"))
    metrics = ExecutionMetrics(
        execution_id="fb-2",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="primary-model",
        start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.workflow_type = WorkflowType.GENERAL_QA
    with pytest.raises(Exception):
        await orchestrator._execute_with_fallback(req, "fb-2", metrics)


# ── _assemble_context ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_assemble_context_with_valid_mode(orchestrator, mock_context_service):
    mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
    req = MagicMock()
    req.prompt = "test"
    req.repo_path = "/tmp"
    req.workflow_type = WorkflowType.GENERAL_QA
    req.context_mode = "balanced"
    req.max_tokens = None
    req.include_files = None
    result = await orchestrator._assemble_context(req)
    assert result.system_prompt == "System prompt"


@pytest.mark.asyncio
async def test_assemble_context_with_invalid_mode_falls_back_to_balanced(orchestrator, mock_context_service):
    mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
    req = MagicMock()
    req.prompt = "test"
    req.repo_path = "/tmp"
    req.workflow_type = WorkflowType.GENERAL_QA
    req.context_mode = "invalid_mode_xyz"
    req.max_tokens = None
    req.include_files = None
    result = await orchestrator._assemble_context(req)
    assert result is not None


# ── _store_artifacts error path ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_artifacts_handles_error_gracefully(orchestrator, mock_artifact_store):
    from app.services.orchestrator import ExecutionMetrics
    mock_artifact_store.save_artifact = AsyncMock(side_effect=Exception("storage failed"))
    metrics = ExecutionMetrics(
        execution_id="art-err",
        workflow_type=WorkflowType.GENERAL_QA,
        model_used="m",
        start_time=datetime.utcnow(),
    )
    metrics.response_tokens = 0
    req = MagicMock()
    req.workflow_type = WorkflowType.GENERAL_QA
    req.prompt = "test"
    # Should not raise — error is caught and logged
    result = await orchestrator._store_artifacts(
        execution_id="art-err",
        request=req,
        response="Some response",
        context=_make_mock_context(),
        metrics=metrics,
    )
    assert isinstance(result, list)
    assert len(result) == 0


# ── get_orchestrator factory ───────────────────────────────────────────────────

def test_get_orchestrator_returns_singleton():
    from app.services.orchestrator import get_orchestrator
    with patch("app.services.orchestrator.get_default_client"), \
         patch("app.services.orchestrator.get_model_service"), \
         patch("app.services.orchestrator.get_context_service"), \
         patch("app.services.orchestrator.get_artifact_store"), \
         patch("app.services.orchestrator.get_default_parser"), \
         patch("app.services.orchestrator.get_skillfile_client"), \
         patch("app.services.orchestrator.AgentFactory"), \
         patch("app.services.orchestrator.DebugWorkflow"), \
         patch("app.services.orchestrator.RefactorWorkflow"), \
         patch("app.services.orchestrator.TestingWorkflow"), \
         patch("app.services.orchestrator._default_orchestrator", None):
        import app.services.orchestrator as orch_module
        orch_module._default_orchestrator = None
        o1 = get_orchestrator()
        assert o1 is not None
        orch_module._default_orchestrator = None  # reset for other tests


# ── execute_streaming ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_streaming_yields_tokens(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    from app.utils.validators import PathValidator

    async def fake_chat_stream(**kwargs):
        for token in ["Hello", " ", "world"]:
            yield token

    mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
    mock_model_service.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
    mock_ollama.chat = fake_chat_stream

    req = MagicMock()
    req.workflow_type = WorkflowType.GENERAL_QA
    req.prompt = "test"
    req.preferred_model = None
    req.temperature = 0.2
    req.max_tokens = None
    req.metadata = {}
    req.repo_path = str(tmp_path)

    tokens = []
    async for token in orchestrator.execute_streaming(req):
        tokens.append(token)
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_execute_streaming_with_preferred_model(orchestrator, mock_context_service, mock_ollama):
    async def fake_stream(**kwargs):
        yield "token1"
        yield "token2"

    mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
    mock_ollama.chat = fake_stream

    req = MagicMock()
    req.workflow_type = WorkflowType.GENERAL_QA
    req.prompt = "test"
    req.preferred_model = "llama3:8b"
    req.temperature = 0.2
    req.max_tokens = None
    req.metadata = {}
    req.repo_path = "/tmp"

    tokens = []
    async for token in orchestrator.execute_streaming(req):
        tokens.append(token)
    assert "token1" in tokens


@pytest.mark.asyncio
async def test_execute_streaming_error_yields_error_message(orchestrator, mock_context_service, mock_ollama):
    mock_context_service.assemble_context = AsyncMock(side_effect=Exception("ctx error"))

    req = MagicMock()
    req.workflow_type = WorkflowType.GENERAL_QA
    req.prompt = "test"
    req.preferred_model = "llama3:8b"
    req.temperature = 0.2
    req.max_tokens = None
    req.metadata = {}
    req.repo_path = "/tmp"

    tokens = []
    async for token in orchestrator.execute_streaming(req):
        tokens.append(token)
    assert any("[Error:" in t for t in tokens)


# ── _execute_workflow_sync ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_workflow_sync_dispatches_to_single_agent(orchestrator, mock_context_service, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="result")
    metrics = ExecutionMetrics(
        execution_id="ws-1",
        workflow_type=WorkflowType.CODE_REVIEW,
        model_used="m",
        start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.workflow_type = WorkflowType.CODE_REVIEW
    req.temperature = 0.2
    req.max_tokens = None
    ctx = _make_mock_context()
    result = await orchestrator._execute_workflow_sync(req, ctx, "qwen2.5-coder:7b", metrics)
    assert result == "result"


@pytest.mark.asyncio
async def test_execute_workflow_sync_multi_agent_branch(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    # Patch _execute_refactoring_workflow to be called
    mock_refactor = AsyncMock(return_value="refactored result")
    orchestrator._multi_agent_workflows[WorkflowType.CODE_REFACTORING] = mock_refactor

    metrics = ExecutionMetrics(
        execution_id="ws-2",
        workflow_type=WorkflowType.CODE_REFACTORING,
        model_used="m",
        start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.workflow_type = WorkflowType.CODE_REFACTORING
    ctx = _make_mock_context()
    result = await orchestrator._execute_workflow_sync(req, ctx, "m", metrics)
    assert result == "refactored result"
    mock_refactor.assert_called_once()


# ── individual workflow executors ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_code_review_calls_single_agent(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="Review done")
    metrics = ExecutionMetrics(
        execution_id="cr-1", workflow_type=WorkflowType.CODE_REVIEW,
        model_used="m", start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.temperature = 0.2
    req.max_tokens = None
    result = await orchestrator._execute_code_review(req, _make_mock_context(), "m", metrics)
    assert result == "Review done"


@pytest.mark.asyncio
async def test_execute_code_refactoring_calls_single_agent(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="Refactored")
    metrics = ExecutionMetrics(
        execution_id="rf-1", workflow_type=WorkflowType.CODE_REFACTORING,
        model_used="m", start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.temperature = 0.2
    req.max_tokens = None
    result = await orchestrator._execute_code_refactoring(req, _make_mock_context(), "m", metrics)
    assert result == "Refactored"


@pytest.mark.asyncio
async def test_execute_debug_analysis_calls_single_agent(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="Debug analysis")
    metrics = ExecutionMetrics(
        execution_id="da-1", workflow_type=WorkflowType.DEBUG_ANALYSIS,
        model_used="m", start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.temperature = 0.2
    req.max_tokens = None
    result = await orchestrator._execute_debug_analysis(req, _make_mock_context(), "m", metrics)
    assert result == "Debug analysis"


@pytest.mark.asyncio
async def test_execute_architecture_analysis_calls_single_agent(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="Architecture analysis")
    metrics = ExecutionMetrics(
        execution_id="aa-1", workflow_type=WorkflowType.ARCHITECTURE_ANALYSIS,
        model_used="m", start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.temperature = 0.2
    req.max_tokens = None
    result = await orchestrator._execute_architecture_analysis(req, _make_mock_context(), "m", metrics)
    assert result == "Architecture analysis"


@pytest.mark.asyncio
async def test_execute_test_generation_calls_single_agent(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="Tests generated")
    metrics = ExecutionMetrics(
        execution_id="tg-1", workflow_type=WorkflowType.TEST_GENERATION,
        model_used="m", start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.temperature = 0.2
    req.max_tokens = None
    result = await orchestrator._execute_test_generation(req, _make_mock_context(), "m", metrics)
    assert result == "Tests generated"


@pytest.mark.asyncio
async def test_execute_documentation_calls_single_agent(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="Docs written")
    metrics = ExecutionMetrics(
        execution_id="doc-1", workflow_type=WorkflowType.DOCUMENTATION,
        model_used="m", start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.temperature = 0.2
    req.max_tokens = None
    result = await orchestrator._execute_documentation(req, _make_mock_context(), "m", metrics)
    assert result == "Docs written"


@pytest.mark.asyncio
async def test_execute_impact_analysis_calls_single_agent(orchestrator, mock_ollama):
    from app.services.orchestrator import ExecutionMetrics
    mock_ollama.chat = AsyncMock(return_value="Impact analysis")
    metrics = ExecutionMetrics(
        execution_id="ia-1", workflow_type=WorkflowType.IMPACT_ANALYSIS,
        model_used="m", start_time=datetime.utcnow(),
    )
    req = MagicMock()
    req.temperature = 0.2
    req.max_tokens = None
    result = await orchestrator._execute_impact_analysis(req, _make_mock_context(), "m", metrics)
    assert result == "Impact analysis"


# ── ensure_fresh_intelligence skills branch ────────────────────────────────────

@pytest.mark.asyncio
async def test_ensure_fresh_intelligence_with_installed_skillfile(orchestrator, tmp_path):
    orchestrator.skillfile = MagicMock()
    orchestrator.skillfile.is_installed = True
    with patch.object(orchestrator, "_auto_update_graphify", new=AsyncMock(
        return_value={"updated": False, "version": None}
    )), patch.object(orchestrator, "_auto_update_skills", new=AsyncMock(
        return_value={"updated": True, "version": "2024-01-01"}
    )):
        result = await orchestrator.ensure_fresh_intelligence(str(tmp_path))
    assert result["skills_updated"] is True


@pytest.mark.asyncio
async def test_ensure_fresh_intelligence_skills_error_handled(orchestrator, tmp_path):
    orchestrator.skillfile = MagicMock()
    orchestrator.skillfile.is_installed = True
    with patch.object(orchestrator, "_auto_update_graphify", new=AsyncMock(
        return_value={"updated": False, "version": None}
    )), patch.object(orchestrator, "_auto_update_skills", new=AsyncMock(
        side_effect=Exception("skills error")
    )):
        result = await orchestrator.ensure_fresh_intelligence(str(tmp_path))
    assert result["skills_updated"] is False


# ── _auto_update_graphify ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_update_graphify_no_output_dir(orchestrator, tmp_path):
    with patch("app.services.orchestrator.GraphifyWrapper") as mock_gw:
        mock_instance = AsyncMock()
        mock_gw.return_value = mock_instance
        mock_instance.run_graphify = AsyncMock()
        orchestrator.context_service.clear_cache = MagicMock()
        result = await orchestrator._auto_update_graphify(str(tmp_path))
    assert result["updated"] is True


@pytest.mark.asyncio
async def test_auto_update_graphify_fresh_graph_no_update(orchestrator, tmp_path):
    import time as time_mod
    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    graph_file = graphify_dir / "graph.json"
    graph_file.write_text("{}")
    # Touch it so it's fresh (< 24h old)
    orchestrator.context_service.clear_cache = MagicMock()
    with patch.object(orchestrator, "_has_code_changed_since", new=AsyncMock(return_value=False)):
        result = await orchestrator._auto_update_graphify(str(tmp_path))
    assert result["updated"] is False


@pytest.mark.asyncio
async def test_auto_update_graphify_force_refreshes(orchestrator, tmp_path):
    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    graph_file = graphify_dir / "graph.json"
    graph_file.write_text("{}")
    orchestrator.context_service.clear_cache = MagicMock()
    with patch("app.services.orchestrator.GraphifyWrapper") as mock_gw:
        mock_instance = AsyncMock()
        mock_gw.return_value = mock_instance
        mock_instance.run_graphify = AsyncMock()
        result = await orchestrator._auto_update_graphify(str(tmp_path), force=True)
    assert result["updated"] is True


# ── _auto_update_skills ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_update_skills_force_with_outdated(orchestrator, tmp_path):
    orchestrator.skillfile = AsyncMock()
    orchestrator.skillfile.get_status = AsyncMock(return_value={"outdated": 2})
    orchestrator.skillfile.install_skills = AsyncMock()
    orchestrator.context_service.clear_cache = MagicMock()
    result = await orchestrator._auto_update_skills(force=True)
    assert result["updated"] is True
    assert result["skills_updated"] == 2


@pytest.mark.asyncio
async def test_auto_update_skills_not_needed(orchestrator):
    result = await orchestrator._auto_update_skills(force=False)
    assert result["updated"] is False


# ── _has_code_changed_since ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_has_code_changed_since_no_graph_file(orchestrator, tmp_path):
    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    result = await orchestrator._has_code_changed_since(tmp_path, graphify_dir)
    assert result is True


@pytest.mark.asyncio
async def test_has_code_changed_since_with_changed_files(orchestrator, tmp_path):
    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    graph_file = graphify_dir / "graph.json"
    graph_file.write_text("{}")

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"app/main.py\napp/utils.py\n", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await orchestrator._has_code_changed_since(tmp_path, graphify_dir)
    assert result is True


@pytest.mark.asyncio
async def test_has_code_changed_since_no_code_changes(orchestrator, tmp_path):
    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    graph_file = graphify_dir / "graph.json"
    graph_file.write_text("{}")

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"README.md\n", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await orchestrator._has_code_changed_since(tmp_path, graphify_dir)
    assert result is False


@pytest.mark.asyncio
async def test_has_code_changed_since_git_error(orchestrator, tmp_path):
    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    graph_file = graphify_dir / "graph.json"
    graph_file.write_text("{}")

    with patch("asyncio.create_subprocess_exec", side_effect=Exception("git not found")):
        result = await orchestrator._has_code_changed_since(tmp_path, graphify_dir)
    assert result is False


# ── execute with CircuitBreakerOpenError ───────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_circuit_breaker_open_uses_fallback(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    from app.utils.retry import CircuitBreakerOpenError
    from app.utils.validators import PathValidator
    (tmp_path / ".git").mkdir()
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    PathValidator.set_allowed_base_dirs([tmp_path])
    try:
        mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
        mock_model_service.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
        # First execute_workflow_sync raises CircuitBreakerOpenError, fallback succeeds
        fallback_response = OrchestratorResponse(
            execution_id="fb",
            status=ExecutionStatus.FALLBACK_USED,
            workflow_type=WorkflowType.GENERAL_QA,
            model_used="fallback",
            response="fallback result",
        )
        with patch.object(
            orchestrator, "_execute_workflow_sync",
            new=AsyncMock(side_effect=CircuitBreakerOpenError("circuit open"))
        ), patch.object(
            orchestrator, "_execute_with_fallback",
            new=AsyncMock(return_value=fallback_response)
        ), patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
            return_value={"graphify_updated": False, "skills_updated": False}
        )):
            req = OrchestratorRequest(
                repo_path=str(tmp_path),
                prompt="test",
                workflow_type=WorkflowType.GENERAL_QA,
            )
            result = await orchestrator.execute(req)
        assert result.status == ExecutionStatus.FALLBACK_USED
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


# ── execution history trimming ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execution_history_trimmed_at_max(orchestrator, mock_context_service, mock_model_service, mock_ollama, tmp_path):
    from app.utils.validators import PathValidator
    from app.services.orchestrator import ExecutionMetrics
    (tmp_path / ".git").mkdir()
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    PathValidator.set_allowed_base_dirs([tmp_path])
    # Pre-fill history to max
    orchestrator._max_history = 3
    for i in range(3):
        m = ExecutionMetrics(
            execution_id=f"old-{i}",
            workflow_type=WorkflowType.GENERAL_QA,
            model_used="m",
            start_time=datetime.utcnow(),
            status=ExecutionStatus.COMPLETED,
        )
        orchestrator._execution_history.append(m)
    try:
        mock_context_service.assemble_context = AsyncMock(return_value=_make_mock_context())
        mock_model_service.select_model = AsyncMock(return_value="qwen2.5-coder:7b")
        mock_ollama.chat = AsyncMock(return_value="response")
        with patch.object(orchestrator, "ensure_fresh_intelligence", new=AsyncMock(
            return_value={"graphify_updated": False, "skills_updated": False}
        )):
            req = OrchestratorRequest(
                repo_path=str(tmp_path),
                prompt="test",
                workflow_type=WorkflowType.GENERAL_QA,
            )
            await orchestrator.execute(req)
        assert len(orchestrator._execution_history) <= orchestrator._max_history
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig
