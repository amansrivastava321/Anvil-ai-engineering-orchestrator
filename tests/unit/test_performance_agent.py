"""Tests for app.agents.specialized.performance_agent — PerformanceAgent."""
import pytest
from unittest.mock import AsyncMock
from app.agents.specialized.performance_agent import PerformanceAgent
from app.agents.base_agent import AgentTask, AgentResult, AgentStatus


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="Performance analysis: no major bottlenecks found.")
    return client


@pytest.fixture
def agent(mock_ollama):
    return PerformanceAgent(ollama_client=mock_ollama)


@pytest.fixture
def task(tmp_path):
    (tmp_path / "app.py").write_text("def process(items):\n    return [x * 2 for x in items]\n")
    return AgentTask(
        prompt="Find performance issues in this codebase",
        repo_path=str(tmp_path),
        model="qwen2.5-coder:7b",
    )


def test_performance_agent_name(agent):
    assert agent.name == "performance"


def test_performance_agent_agent_name(agent):
    assert agent.agent_name == "performance_agent"


def test_performance_agent_description(agent):
    assert len(agent.description) > 10


def test_performance_agent_system_prompt(agent):
    assert "performance" in agent.system_prompt.lower()


@pytest.mark.asyncio
async def test_performance_agent_run_returns_completed(agent, task):
    result = await agent.run(task)
    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.COMPLETED
    assert result.response != ""


@pytest.mark.asyncio
async def test_performance_agent_creates_tools(tmp_path, agent):
    tools = agent._create_tools(str(tmp_path))
    tool_names = [t.name for t in tools]
    assert "read_file" in tool_names
    assert "analyze_dependencies" in tool_names


@pytest.mark.asyncio
async def test_performance_agent_handles_error(mock_ollama, task):
    mock_ollama.chat.side_effect = RuntimeError("model down")
    agent = PerformanceAgent(ollama_client=mock_ollama)
    result = await agent.run(task)
    assert result.status == AgentStatus.FAILED
    assert "model down" in (result.error or "")
