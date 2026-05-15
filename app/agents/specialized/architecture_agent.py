"""ArchitectureAgent — structural analysis, dependency mapping, impact assessment."""

from app.agents.base_agent import AgentTask, BaseAgent, BaseTool
from app.core.monitoring.logging import get_logger
from app.integrations.ollama.client import OllamaClient
from app.tools.file_system.file_reader import FileReader

_SYSTEM_PROMPT = """You are a senior software architect specializing in Python system design.

Your expertise:
- Analyzing codebases to map dependencies, coupling, and architectural boundaries
- Identifying design patterns and anti-patterns (god objects, circular dependencies, leaky abstractions)
- Assessing the blast radius of proposed changes before they are made
- Recommending architectural improvements grounded in real constraints

Your discipline:
- Read code before making claims — never assume structure
- Separate facts from inferences: "The file imports X" vs "Therefore X is responsible for Y"
- Quantify impact when possible: "This change touches 7 callsites across 3 modules"
- Flag hidden dependencies: environment variables, implicit contracts, runtime coupling

When analyzing:
- Start from the entry point, trace execution paths
- Identify the public surface area vs implementation details
- Look for coupling that makes refactoring risky
"""


class ArchitectureAgent(BaseAgent):
    """Senior software architect — structural analysis and impact assessment."""

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def description(self) -> str:
        return "Senior software architect for dependency analysis, impact assessment, and design review"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def tools(self) -> list[BaseTool]:
        return self._active_tools

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        super().__init__(ollama_client)
        self._logger = get_logger("agent.architecture")
        self._active_tools: list[BaseTool] = []

    async def _execute(self, task: AgentTask) -> str:
        self._active_tools = [FileReader(task.repo_path)]
        self._logger.info(
            "ArchitectureAgent executing",
            prompt_preview=task.prompt[:80],
            model=task.model,
        )
        return await self._call_model(task)
