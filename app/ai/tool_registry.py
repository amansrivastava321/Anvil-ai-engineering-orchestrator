"""Tool registry — catalog of tools available to the AI organization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolDefinition:
    name: str
    description: str
    when_to_use: str
    input_schema: Dict[str, str] = field(default_factory=dict)
    output_description: str = ""
    risk_level: str = "low"  # low / medium / high


class ToolRegistry:
    """Registry of tools the AI organization can invoke.

    Tools are declared here so the CEO and council members can reason about
    *what is available* when forming their plans. Actual execution goes through
    the orchestrator's tool layer.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_defaults()

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_all(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def by_risk(self, risk_level: str) -> List[ToolDefinition]:
        return [t for t in self._tools.values() if t.risk_level == risk_level]

    def describe_all(self) -> str:
        """Human-readable catalog for injecting into LLM prompts."""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  When: {t.when_to_use}")
        return "\n".join(lines)

    def _register_defaults(self) -> None:
        defaults = [
            ToolDefinition(
                name="file_reader",
                description="Read source files from the repository",
                when_to_use="Understanding existing code before making changes",
                output_description="File contents as text",
                risk_level="low",
            ),
            ToolDefinition(
                name="file_writer",
                description="Write or modify source files",
                when_to_use="Applying code changes or generating new files",
                risk_level="medium",
            ),
            ToolDefinition(
                name="test_runner",
                description="Execute the test suite and report results",
                when_to_use="Verifying changes do not introduce regressions",
                output_description="Pass/fail counts and failure details",
                risk_level="low",
            ),
            ToolDefinition(
                name="ast_analyzer",
                description="Parse and analyze code structure via AST",
                when_to_use="Understanding code structure, dependencies, call graphs",
                risk_level="low",
            ),
            ToolDefinition(
                name="dependency_analyzer",
                description="Analyze module import graphs and external dependencies",
                when_to_use="Impact analysis before refactoring or architecture changes",
                risk_level="low",
            ),
            ToolDefinition(
                name="security_scanner",
                description="Scan code for security vulnerabilities",
                when_to_use="Security audits, reviewing authentication/authorization code",
                risk_level="low",
            ),
            ToolDefinition(
                name="coverage_analyzer",
                description="Measure and report test coverage",
                when_to_use="Identifying untested code paths before deployment",
                risk_level="low",
            ),
            ToolDefinition(
                name="git_log",
                description="Query git history for recent changes",
                when_to_use="Correlating bugs with recent commits, impact analysis",
                risk_level="low",
            ),
            ToolDefinition(
                name="search_codebase",
                description="Search repository for patterns, symbols, or text",
                when_to_use="Finding all usages of a function or pattern across the codebase",
                risk_level="low",
            ),
        ]
        for t in defaults:
            self._tools[t.name] = t


_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
