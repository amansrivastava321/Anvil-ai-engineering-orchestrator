"""TestingAgent — test strategy, test generation, and coverage analysis."""

from app.agents.base_agent import AgentTask, BaseAgent, BaseTool
from app.core.monitoring.logging import get_logger
from app.integrations.ollama.client import OllamaClient
from app.tools.file_system.file_reader import FileReader
from app.tools.file_system.file_writer import FileWriter
from app.tools.testing.test_runner import TestRunner

_SYSTEM_PROMPT = """You are an expert in software testing and quality assurance for Python systems.

Your expertise:
- Designing test strategies: what to unit-test vs integration-test vs leave to e2e
- Writing effective pytest tests with meaningful assertions
- Identifying what is NOT tested and what gaps pose real risk
- Async testing patterns: pytest-asyncio, event loop scoping, mock strategies

Your discipline:
- Tests must fail before implementation — no retroactive test writing
- Each test tests one behavior, has one logical assertion
- Test names describe the behavior: test_<thing>_<condition>_<expectation>
- Mocks only at system boundaries: external APIs, filesystem, databases
- Avoid testing implementation details — test the public contract
- Don't mock things you own; mock things you don't (third-party clients)

When generating tests:
- Output complete, runnable test files with all imports
- Include fixtures that make tests independent and fast
- Flag slow tests (I/O, subprocess) and add timeout markers
"""


class TestingAgent(BaseAgent):
    """Expert QA engineer — test strategy, test generation, and coverage analysis."""

    @property
    def name(self) -> str:
        return "testing"

    @property
    def description(self) -> str:
        return "Expert QA engineer for test strategy, test generation, and coverage analysis"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def tools(self) -> list[BaseTool]:
        return self._active_tools

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        super().__init__(ollama_client)
        self._logger = get_logger("agent.testing")
        self._active_tools: list[BaseTool] = []

    async def _execute(self, task: AgentTask) -> str:
        self._active_tools = [
            FileReader(task.repo_path),
            FileWriter(task.repo_path),
            TestRunner(task.repo_path),
        ]
        self._logger.info(
            "TestingAgent executing",
            prompt_preview=task.prompt[:80],
            model=task.model,
        )
        return await self._call_model(task)
