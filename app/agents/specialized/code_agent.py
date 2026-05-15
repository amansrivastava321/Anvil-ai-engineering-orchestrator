"""CodeAgent — expert software engineer for code generation, review, and refactoring."""

from app.agents.base_agent import AgentTask, BaseAgent, BaseTool
from app.core.monitoring.logging import get_logger
from app.integrations.ollama.client import OllamaClient
from app.tools.file_system.file_reader import FileReader
from app.tools.file_system.file_writer import FileWriter
from app.tools.testing.test_runner import TestRunner

_SYSTEM_PROMPT = """You are an expert software engineer specializing in Python development.

Your core discipline:
- Write clean, type-safe, async Python (3.11+) with full type annotations
- Follow TDD: write tests first, implement minimal code to make them pass
- No dead code, no placeholders, no TODOs in production code
- Functions do one thing; files have one responsibility
- Errors surface explicitly — never swallow exceptions silently
- Prefer built-in generics (list, dict, str | None) over typing module equivalents

When reviewing code:
- Flag type safety violations and missing annotations
- Identify hidden bugs: race conditions, missing error handling, incorrect async patterns
- Call out unnecessary complexity and premature abstraction

When generating code:
- Output only the code requested — no explanations unless explicitly asked
- Include all necessary imports
- Write tests alongside implementations when generating new modules
"""


class CodeAgent(BaseAgent):
    """Expert software engineer — code generation, review, and refactoring."""

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Expert Python software engineer for code generation, review, and refactoring"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def tools(self) -> list[BaseTool]:
        return self._active_tools

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        super().__init__(ollama_client)
        self._logger = get_logger("agent.code")
        self._active_tools: list[BaseTool] = []

    async def _execute(self, task: AgentTask) -> str:
        self._active_tools = [
            FileReader(task.repo_path),
            FileWriter(task.repo_path),
            TestRunner(task.repo_path),
        ]
        self._logger.info(
            "CodeAgent executing",
            prompt_preview=task.prompt[:80],
            model=task.model,
            repo_path=task.repo_path,
        )
        return await self._call_model(task)
