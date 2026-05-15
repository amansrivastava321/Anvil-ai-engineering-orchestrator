# tests/unit/test_agents.py
import pytest
from unittest.mock import AsyncMock
from app.agents.base_agent import AgentTask, AgentResult, AgentStatus
from app.agents.specialized.code_agent import CodeAgent


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="def hello(): return 'world'")
    return client


@pytest.fixture
def task():
    return AgentTask(
        prompt="Write a hello world function",
        repo_path="/tmp",
        model="qwen2.5-coder:7b",
    )


@pytest.mark.asyncio
async def test_code_agent_name():
    agent = CodeAgent(ollama_client=AsyncMock())
    assert agent.name == "code"


@pytest.mark.asyncio
async def test_code_agent_run_returns_completed_result(mock_ollama, task):
    agent = CodeAgent(ollama_client=mock_ollama)
    result = await agent.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert "hello" in result.response


@pytest.mark.asyncio
async def test_code_agent_registers_tools(mock_ollama, task):
    agent = CodeAgent(ollama_client=mock_ollama)
    await agent.run(task)
    tool_names = [t.name for t in agent.tools]
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "run_tests" in tool_names


@pytest.mark.asyncio
async def test_code_agent_system_prompt_contains_python_discipline(mock_ollama):
    agent = CodeAgent(ollama_client=mock_ollama)
    prompt = agent.system_prompt
    assert "Python" in prompt
    assert "TDD" in prompt or "test" in prompt.lower()


# --- ArchitectureAgent tests ---

from app.agents.specialized.architecture_agent import ArchitectureAgent


@pytest.mark.asyncio
async def test_architecture_agent_name():
    agent = ArchitectureAgent(ollama_client=AsyncMock())
    assert agent.name == "architecture"


@pytest.mark.asyncio
async def test_architecture_agent_registers_file_reader(mock_ollama, task):
    agent = ArchitectureAgent(ollama_client=mock_ollama)
    await agent.run(task)
    assert any(t.name == "read_file" for t in agent.tools)


@pytest.mark.asyncio
async def test_architecture_agent_system_prompt_covers_analysis(mock_ollama):
    agent = ArchitectureAgent(ollama_client=mock_ollama)
    prompt = agent.system_prompt
    assert "architect" in prompt.lower() or "dependencies" in prompt.lower()


# --- TestingAgent tests ---

from app.agents.specialized.testing_agent import TestingAgent


@pytest.mark.asyncio
async def test_testing_agent_name():
    agent = TestingAgent(ollama_client=AsyncMock())
    assert agent.name == "testing"


@pytest.mark.asyncio
async def test_testing_agent_registers_test_runner(mock_ollama, task):
    agent = TestingAgent(ollama_client=mock_ollama)
    await agent.run(task)
    assert any(t.name == "run_tests" for t in agent.tools)


# --- DocumentationAgent tests ---

from app.agents.specialized.documentation_agent import DocumentationAgent


@pytest.mark.asyncio
async def test_documentation_agent_name():
    agent = DocumentationAgent(ollama_client=AsyncMock())
    assert agent.name == "documentation"


@pytest.mark.asyncio
async def test_documentation_agent_registers_reader_and_writer(mock_ollama, task):
    agent = DocumentationAgent(ollama_client=mock_ollama)
    await agent.run(task)
    tool_names = {t.name for t in agent.tools}
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "run_tests" not in tool_names  # docs agent doesn't run tests


def test_architecture_agent_description():
    agent = ArchitectureAgent(ollama_client=AsyncMock())
    assert isinstance(agent.description, str)
    assert len(agent.description) > 10


def test_code_agent_description():
    from app.agents.specialized.code_agent import CodeAgent
    agent = CodeAgent(ollama_client=AsyncMock())
    assert isinstance(agent.description, str)
    assert len(agent.description) > 10


def test_testing_agent_description():
    agent = TestingAgent(ollama_client=AsyncMock())
    assert isinstance(agent.description, str)
    assert len(agent.description) > 10


def test_documentation_agent_description():
    agent = DocumentationAgent(ollama_client=AsyncMock())
    assert isinstance(agent.description, str)
    assert len(agent.description) > 10
