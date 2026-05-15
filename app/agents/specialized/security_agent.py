"""Security agent — vulnerability scanning, credential detection, audit reporting."""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent, AgentResult, AgentTask, AgentStatus, BaseTool
from app.tools.file_system.file_reader import FileReader
from app.tools.code_analysis.security_scanner import SecurityScanner
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert security engineer and code auditor.

Your mission: identify security vulnerabilities, credential leaks, unsafe patterns,
and compliance issues in code. You produce precise, actionable findings.

You work with these tools:
- read_file: Read source files for detailed review
- scan_security: Automated pattern-based security scanning

For every finding you report:
1. Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
2. Location: exact file + line number
3. Issue: what the vulnerability is
4. Impact: what an attacker could do
5. Fix: specific remediation code/steps

Be systematic. Cover: injection, auth, crypto, secrets, input validation,
dependency risks, and configuration issues."""


class SecurityAgent(BaseAgent):
    """Scans code for security vulnerabilities and produces audit reports."""

    @property
    def agent_name(self) -> str:
        return "security_agent"

    @property
    def name(self) -> str:
        return "security"

    @property
    def description(self) -> str:
        return "Security engineer for vulnerability scanning, credential detection, and audit reporting"

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
        return [FileReader(repo_path), SecurityScanner()]

    async def _execute(self, task: AgentTask) -> str:
        self._active_tools = self._create_tools(task.repo_path)
        execution_id = task.task_id

        scanner = SecurityScanner()
        scan_result = await scanner.execute(path=task.repo_path, recursive=True)

        prompt = (
            f"Security audit request: {task.prompt}\n\n"
            f"Automated scanner found:\n{scan_result.output}\n\n"
            f"Provide a structured security report with prioritized findings and fixes."
        )

        task_with_scan = AgentTask(
            task_id=task.task_id,
            prompt=prompt,
            repo_path=task.repo_path,
            context=task.context,
            model=task.model,
            temperature=task.temperature,
            max_tokens=task.max_tokens,
        )

        return await self._call_model(task_with_scan)
