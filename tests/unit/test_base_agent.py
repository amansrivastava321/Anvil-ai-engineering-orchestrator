# tests/unit/test_base_agent.py
import pytest
from unittest.mock import AsyncMock
from app.agents.base_agent import (
    AgentTask,
    AgentResult,
    AgentStatus,
    BaseAgent,
    ToolResult,
)


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent behaviour."""

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


# --- AgentFactory tests ---

from app.agents.agent_factory import AgentFactory


class AnotherConcreteAgent(BaseAgent):
    """Second concrete agent for factory testing."""

    @property
    def name(self) -> str:
        return "another_agent"

    @property
    def description(self) -> str:
        return "Another test agent"

    @property
    def system_prompt(self) -> str:
        return "You are another test agent."

    async def _execute(self, task: AgentTask) -> str:
        return "another response"


def test_factory_list_agents_returns_all_registered_types():
    factory = AgentFactory()
    agents = factory.list_agents()
    assert "code" in agents
    assert "architecture" in agents
    assert "testing" in agents
    assert "documentation" in agents


def test_factory_get_agent_raises_for_unknown_type():
    factory = AgentFactory()
    with pytest.raises(KeyError, match="Unknown agent type"):
        factory.get_agent("nonexistent")


def test_factory_caches_agent_instances(mock_ollama):
    # Patch the registry with a concrete test agent to avoid importing specialized agents
    import app.agents.agent_factory as factory_module
    original_registry = dict(factory_module._REGISTRY)
    factory_module._REGISTRY["test"] = "tests.unit.test_base_agent:ConcreteAgent"

    try:
        factory = AgentFactory(ollama_client=mock_ollama)
        agent1 = factory.get_agent("test")
        agent2 = factory.get_agent("test")
        assert agent1 is agent2  # same instance returned from cache
    finally:
        factory_module._REGISTRY.clear()
        factory_module._REGISTRY.update(original_registry)


def test_get_agent_factory_singleton(mock_ollama):
    import app.agents.agent_factory as factory_module
    orig = factory_module._factory
    factory_module._factory = None
    try:
        from app.agents.agent_factory import get_agent_factory
        factory = get_agent_factory(ollama_client=mock_ollama)
        assert factory is not None
        # Second call returns same instance
        factory2 = get_agent_factory()
        assert factory is factory2
    finally:
        factory_module._factory = orig


@pytest.mark.asyncio
async def test_run_streaming_exception_propagates(agent, task, mock_ollama):
    async def failing_gen(*args, **kwargs):
        yield "partial"
        raise RuntimeError("model crashed")

    mock_ollama.chat = failing_gen
    tokens = []
    with pytest.raises(RuntimeError, match="model crashed"):
        async for token in agent.run_streaming(task):
            tokens.append(token)
    assert "partial" in tokens


def test_build_messages_with_focus(agent, task):
    messages = agent._build_messages(task, focus="security vulnerabilities")
    system_msg = messages[0]["content"]
    assert "FOCUS: security vulnerabilities" in system_msg


def test_build_messages_with_context_user_prompt(agent, task, mock_ollama):
    from unittest.mock import MagicMock
    ctx = MagicMock()
    ctx.user_prompt = "what is the bug?"
    task.context = ctx
    messages = agent._build_messages(task)
    user_msg = messages[1]["content"]
    assert "what is the bug?" in user_msg
    assert task.prompt in user_msg
