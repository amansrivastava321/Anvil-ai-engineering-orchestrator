import pytest
from unittest.mock import AsyncMock, MagicMock
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
    assert mock_agent.run.call_count == 3


def test_debug_workflow_result_to_markdown():
    result = DebugWorkflowResult(root_cause_analysis="root cause", solution="fix code")
    md = result.to_markdown()
    assert "## Root Cause Analysis" in md
    assert "root cause" in md
    assert "## Solution" in md
    assert "fix code" in md


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
    assert mock_agent.run.call_count == 3


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
    assert mock_agent.run.call_count == 2
