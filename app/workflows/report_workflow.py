"""Report workflow — produce structured engineering reports from agent outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import time

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ReportWorkflowResult:
    title: str
    executive_summary: str
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    full_report: str
    duration_ms: float
    models_used: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}\n",
            "## Executive Summary\n", self.executive_summary, "\n",
            "## Key Findings\n",
        ]
        for i, finding in enumerate(self.findings[:10], 1):
            lines.append(f"{i}. {finding.get('summary', finding)}")
        lines += [
            "\n## Recommendations\n",
        ]
        for r in self.recommendations:
            lines.append(f"- {r}")
        lines += ["\n## Full Report\n", self.full_report]
        return "\n".join(lines)


class ReportWorkflow:
    """Produces structured engineering reports from codebase analysis."""

    async def run(
        self,
        repo_path: str,
        prompt: str,
        context: Any,
        arch_model: str,
        doc_model: str,
        report_title: str = "Engineering Report",
    ) -> ReportWorkflowResult:
        from app.agents.agent_factory import get_agent_factory
        from app.agents.base_agent import AgentTask

        start = time.monotonic()
        factory = get_agent_factory()

        # Step 1: Gather findings via architecture agent
        arch_agent = factory.get_agent("architecture")
        arch_task = AgentTask(
            prompt=f"Analyze this codebase and gather findings for a report: {prompt}",
            repo_path=repo_path,
            context=context,
            model=arch_model,
        )
        arch_result = await arch_agent.run(arch_task)

        # Step 2: Write the report
        doc_agent = factory.get_agent("documentation")
        doc_task = AgentTask(
            prompt=(
                f"Write a structured engineering report titled '{report_title}' based on:\n\n"
                f"{arch_result.response}\n\nOriginal request: {prompt}"
            ),
            repo_path=repo_path,
            context=context,
            model=doc_model,
        )
        doc_result = await doc_agent.run(doc_task)

        # Extract recommendations from response
        recommendations = [
            line.strip().lstrip("-•* ")
            for line in doc_result.response.splitlines()
            if line.strip().startswith(("-", "•", "*", "1.", "2.", "3."))
        ][:10]

        return ReportWorkflowResult(
            title=report_title,
            executive_summary=arch_result.summary or arch_result.response[:300],
            findings=[{"summary": arch_result.summary}],
            recommendations=recommendations or ["Review findings and prioritize actions"],
            full_report=doc_result.response,
            duration_ms=(time.monotonic() - start) * 1000,
            models_used=[arch_model, doc_model],
        )
