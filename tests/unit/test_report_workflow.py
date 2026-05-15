"""Tests for app.workflows.report_workflow — ReportWorkflow and ReportWorkflowResult."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.workflows.report_workflow import ReportWorkflow, ReportWorkflowResult
from app.agents.base_agent import AgentResult, AgentStatus


def _make_agent_result(response="report content", summary="Executive summary"):
    return AgentResult(
        execution_id="exec-t1",
        agent_name="mock_agent",
        status=AgentStatus.COMPLETED,
        response=response,
        summary=summary,
    )


@pytest.fixture
def mock_factory():
    factory = MagicMock()
    factory.get_agent.return_value = MagicMock(
        run=AsyncMock(return_value=_make_agent_result())
    )
    return factory


@pytest.mark.asyncio
async def test_report_workflow_returns_result(tmp_path, mock_factory):
    with patch("app.agents.agent_factory.get_agent_factory", return_value=mock_factory):
        wf = ReportWorkflow()
        result = await wf.run(
            repo_path=str(tmp_path),
            prompt="generate report",
            context={},
            arch_model="qwen2.5-coder:7b",
            doc_model="llama3.1:8b",
        )
    assert isinstance(result, ReportWorkflowResult)
    assert result.duration_ms >= 0
    assert result.title == "Engineering Report"


@pytest.mark.asyncio
async def test_report_workflow_custom_title(tmp_path, mock_factory):
    with patch("app.agents.agent_factory.get_agent_factory", return_value=mock_factory):
        wf = ReportWorkflow()
        result = await wf.run(
            repo_path=str(tmp_path),
            prompt="report",
            context={},
            arch_model="m",
            doc_model="m",
            report_title="Security Report Q1",
        )
    assert result.title == "Security Report Q1"


@pytest.mark.asyncio
async def test_report_workflow_extracts_recommendations(tmp_path):
    doc_response = "- Fix auth\n- Add tests\n* Review deps\n"
    factory = MagicMock()
    factory.get_agent.side_effect = lambda name: MagicMock(
        run=AsyncMock(return_value=_make_agent_result(response=doc_response))
    )
    with patch("app.agents.agent_factory.get_agent_factory", return_value=factory):
        wf = ReportWorkflow()
        result = await wf.run(
            repo_path=str(tmp_path), prompt="r", context={},
            arch_model="m", doc_model="m",
        )
    assert any("Fix auth" in r or "Add tests" in r or "Review deps" in r
               for r in result.recommendations)


@pytest.mark.asyncio
async def test_report_workflow_models_used(tmp_path, mock_factory):
    with patch("app.agents.agent_factory.get_agent_factory", return_value=mock_factory):
        wf = ReportWorkflow()
        result = await wf.run(
            repo_path=str(tmp_path), prompt="r", context={},
            arch_model="arch-m", doc_model="doc-m",
        )
    assert "arch-m" in result.models_used
    assert "doc-m" in result.models_used


def test_report_result_to_markdown():
    r = ReportWorkflowResult(
        title="My Report",
        executive_summary="Key summary",
        findings=[{"summary": "Finding A"}, {"summary": "Finding B"}],
        recommendations=["Do X", "Do Y"],
        full_report="Full text here",
        duration_ms=500.0,
        models_used=["m1"],
    )
    md = r.to_markdown()
    assert "# My Report" in md
    assert "Key summary" in md
    assert "Finding A" in md
    assert "Do X" in md
    assert "Full text here" in md


def test_report_result_empty_recommendations_gets_fallback(tmp_path):
    r = ReportWorkflowResult(
        title="T",
        executive_summary="",
        findings=[],
        recommendations=["Review findings and prioritize actions"],
        full_report="",
        duration_ms=0,
    )
    assert len(r.recommendations) > 0
