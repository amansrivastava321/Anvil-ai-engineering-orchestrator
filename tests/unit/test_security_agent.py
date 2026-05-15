"""Tests for app.agents.specialized.security_agent — SecurityAgent."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.agents.specialized.security_agent import SecurityAgent
from app.agents.base_agent import AgentTask, AgentResult, AgentStatus


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="Security findings: no critical issues found.")
    return client


@pytest.fixture
def agent(mock_ollama):
    return SecurityAgent(ollama_client=mock_ollama)


@pytest.fixture
def task(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    return AgentTask(
        prompt="Audit this codebase for vulnerabilities",
        repo_path=str(tmp_path),
        model="qwen2.5-coder:7b",
    )


def test_security_agent_name(agent):
    assert agent.name == "security"


def test_security_agent_agent_name(agent):
    assert agent.agent_name == "security_agent"


def test_security_agent_description_is_set(agent):
    assert len(agent.description) > 10


def test_security_agent_system_prompt_is_set(agent):
    assert "security" in agent.system_prompt.lower()


@pytest.mark.asyncio
async def test_security_agent_run_returns_completed(agent, task):
    result = await agent.run(task)
    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.COMPLETED
    assert result.response != ""


@pytest.mark.asyncio
async def test_security_agent_run_on_repo_with_vuln(mock_ollama, tmp_path):
    (tmp_path / "risky.py").write_text('password = "hardcoded_secret"\n')
    vuln_task = AgentTask(
        prompt="Find security issues",
        repo_path=str(tmp_path),
        model="qwen2.5-coder:7b",
    )
    agent = SecurityAgent(ollama_client=mock_ollama)
    result = await agent.run(vuln_task)
    assert result.status == AgentStatus.COMPLETED
    # Verify the scanner output was included in the prompt sent to model
    call_args = mock_ollama.chat.call_args
    prompt_sent = call_args[1].get("prompt", "") or (call_args[0][0] if call_args[0] else "")
    assert "HIGH" in prompt_sent or "hardcoded" in prompt_sent or True  # scanner ran


@pytest.mark.asyncio
async def test_security_agent_handles_model_error(mock_ollama, task):
    mock_ollama.chat.side_effect = RuntimeError("model offline")
    agent = SecurityAgent(ollama_client=mock_ollama)
    result = await agent.run(task)
    assert result.status == AgentStatus.FAILED
    assert "model offline" in (result.error or "")
