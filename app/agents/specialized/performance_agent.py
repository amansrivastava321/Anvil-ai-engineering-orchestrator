"""Performance agent — profile hotspots, complexity analysis, optimization guidance."""
from __future__ import annotations

from app.agents.base_agent import BaseAgent, AgentResult, AgentTask, AgentStatus, BaseTool
from app.tools.file_system.file_reader import FileReader
from app.tools.code_analysis.dependency_analyzer import DependencyAnalyzer
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a performance engineering expert.

Your mission: identify performance bottlenecks, inefficient algorithms,
memory leaks, blocking I/O in async code, and optimization opportunities.

You work with these tools:
- read_file: Read source files for analysis
- analyze_dependencies: Understand module coupling and complexity

For every performance issue you find:
1. Location: file + function
2. Issue: what the bottleneck is (O(n²) loop, blocking call, memory leak, etc.)
3. Impact: estimated performance cost
4. Fix: specific optimized implementation

Focus on: algorithmic complexity, I/O patterns, caching opportunities,
async/await correctness, database query patterns, and memory allocation."""


class PerformanceAgent(BaseAgent):
    """Analyzes code for performance issues and suggests optimizations."""

    @property
    def agent_name(self) -> str:
        return "performance_agent"

    @property
    def name(self) -> str:
        return "performance"

    @property
    def description(self) -> str:
        return "Performance engineer for profiling hotspots, complexity analysis, and optimization guidance"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def tools(self) -> list[BaseTool]:
        return self._active_tools

    def __init__(self, ollama_client=None) -> None:
        super().__init__(ollama_client)
        self._active_tools: list[BaseTool] = []

    def _create_tools(self, repo_path: str = "") -> list[BaseTool]:
        return [FileReader(repo_path), DependencyAnalyzer()]

    async def _execute(self, task: AgentTask) -> str:
        self._active_tools = self._create_tools(task.repo_path)

        return await self._call_model(task)
