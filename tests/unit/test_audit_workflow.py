"""Tests for app.workflows.audit_workflow — AuditWorkflow and AuditWorkflowResult."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.workflows.audit_workflow import AuditWorkflow, AuditWorkflowResult
from app.agents.base_agent import AgentResult, AgentStatus


def _make_agent_result(response="analysis output", details=None):
    return AgentResult(
        execution_id="exec-t1",
        agent_name="mock_agent",
        status=AgentStatus.COMPLETED,
        response=response,
        summary="Summary",
        details=details or {},
    )


@pytest.fixture
def mock_factory():
    factory = MagicMock()
    factory.get_agent.return_value = MagicMock(
        run=AsyncMock(return_value=_make_agent_result())
    )
    return factory


@pytest.mark.asyncio
async def test_audit_workflow_returns_result(tmp_path, mock_factory):
    with patch("app.agents.agent_factory.get_agent_factory", return_value=mock_factory), \
         patch("app.agents.agent_factory.AgentFactory.get_agent", return_value=MagicMock(run=AsyncMock(return_value=_make_agent_result()))):
        with patch("app.tools.code_analysis.dependency_analyzer.DependencyAnalyzer") as MockDep:
            mock_dep_inst = MagicMock()
            mock_dep_inst.execute = AsyncMock(return_value=MagicMock(output="dep output"))
            MockDep.return_value = mock_dep_inst
            wf = AuditWorkflow()
            mock_factory.get_agent.return_value = MagicMock(
                run=AsyncMock(return_value=_make_agent_result())
            )
            with patch("app.agents.agent_factory.get_agent_factory", return_value=mock_factory):
                result = await wf.run(
                    repo_path=str(tmp_path),
                    prompt="audit this",
                    context={},
                    arch_model="qwen2.5-coder:7b",
                    security_model="qwen2.5-coder:7b",
                    doc_model="llama3.1:8b",
                )
    assert isinstance(result, AuditWorkflowResult)
    assert result.risk_score in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_audit_workflow_risk_score_low_on_no_findings(tmp_path, mock_factory):
    mock_factory.get_agent.return_value = MagicMock(
        run=AsyncMock(return_value=_make_agent_result())
    )
    with patch("app.agents.agent_factory.get_agent_factory", return_value=mock_factory):
        wf = AuditWorkflow()
        result = await wf.run(
            repo_path=str(tmp_path),
            prompt="test",
            context={},
            arch_model="m",
            security_model="m",
            doc_model="m",
        )
    assert result.risk_score == "LOW"


@pytest.mark.asyncio
async def test_audit_workflow_risk_score_critical_on_critical_findings(tmp_path):
    findings = [{"severity": "CRITICAL", "file": "a.py", "line": 1, "description": "key"}]
    sec_result = _make_agent_result(details={"findings": findings})
    factory = MagicMock()
    arch_agent = MagicMock(run=AsyncMock(return_value=_make_agent_result()))
    sec_agent = MagicMock(run=AsyncMock(return_value=sec_result))
    doc_agent = MagicMock(run=AsyncMock(return_value=_make_agent_result()))
    factory.get_agent.side_effect = lambda name: {
        "architecture": arch_agent,
        "security": sec_agent,
        "documentation": doc_agent,
    }[name]
    with patch("app.agents.agent_factory.get_agent_factory", return_value=factory):
        wf = AuditWorkflow()
        result = await wf.run(
            repo_path=str(tmp_path), prompt="x", context={},
            arch_model="m", security_model="m", doc_model="m",
        )
    assert result.risk_score == "CRITICAL"


def test_audit_result_to_markdown_no_findings():
    r = AuditWorkflowResult(
        architecture_summary="arch",
        security_findings=[],
        dependency_health="clean",
        test_coverage_summary="30%",
        audit_report="full report",
        duration_ms=100.0,
        risk_score="LOW",
    )
    md = r.to_markdown()
    assert "# System Audit Report" in md
    assert "LOW" in md
    assert "No security findings" in md


def test_audit_result_to_markdown_with_findings():
    findings = [{"severity": "HIGH", "file": "x.py", "line": 5, "description": "eval"}]
    r = AuditWorkflowResult(
        architecture_summary="arch",
        security_findings=findings,
        dependency_health="ok",
        test_coverage_summary="50%",
        audit_report="report",
        duration_ms=200.0,
        risk_score="HIGH",
    )
    md = r.to_markdown()
    assert "[HIGH]" in md
    assert "x.py:5" in md


def test_audit_result_models_used():
    r = AuditWorkflowResult(
        architecture_summary="",
        security_findings=[],
        dependency_health="",
        test_coverage_summary="",
        audit_report="",
        duration_ms=0,
        models_used=["m1", "m2"],
    )
    assert "m1" in r.models_used
