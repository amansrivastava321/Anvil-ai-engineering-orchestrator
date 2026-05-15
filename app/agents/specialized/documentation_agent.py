"""DocumentationAgent — docstrings, READMEs, API docs, and changelogs."""

from app.agents.base_agent import AgentTask, BaseAgent, BaseTool
from app.core.monitoring.logging import get_logger
from app.integrations.ollama.client import OllamaClient
from app.tools.file_system.file_reader import FileReader
from app.tools.file_system.file_writer import FileWriter

_SYSTEM_PROMPT = """You are a technical writer and documentation engineer for Python projects.

Your expertise:
- Writing docstrings that explain WHY and WHEN to use a component, not just WHAT it does
- Structuring READMEs for quick orientation and deep reference
- Documenting APIs so consumers don't need to read source code
- Writing changelogs that explain impact, not just changes

Your discipline:
- Read the code before writing about it — no hallucinated behavior
- Match documentation style to existing patterns in the repo
- Docstrings go on public interfaces only — internal helpers rarely need them
- Format: one-line summary, blank line, extended description (if needed)
- Never document WHAT is obvious from the name — document the non-obvious constraints

When writing:
- Start with the most important information (inverted pyramid)
- Use concrete examples over abstract descriptions
- Flag deprecated behavior and migration paths explicitly
"""


class DocumentationAgent(BaseAgent):
    """Technical writer — docstrings, READMEs, API docs, and changelogs."""

    @property
    def name(self) -> str:
        return "documentation"

    @property
    def description(self) -> str:
        return "Technical writer for docstrings, READMEs, API documentation, and changelogs"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def tools(self) -> list[BaseTool]:
        return self._active_tools

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        super().__init__(ollama_client)
        self._logger = get_logger("agent.documentation")
        self._active_tools: list[BaseTool] = []

    async def _execute(self, task: AgentTask) -> str:
        self._active_tools = [
            FileReader(task.repo_path),
            FileWriter(task.repo_path),
        ]
        self._logger.info(
            "DocumentationAgent executing",
            prompt_preview=task.prompt[:80],
            model=task.model,
        )
        return await self._call_model(task)
