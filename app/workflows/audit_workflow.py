"""Full system audit workflow — security + architecture + test coverage + deps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AuditWorkflowResult:
    architecture_summary: str
    security_findings: List[Dict[str, Any]]
    dependency_health: str
    test_coverage_summary: str
    audit_report: str
    duration_ms: float
    models_used: List[str] = field(default_factory=list)
    risk_score: str = "UNKNOWN"  # LOW / MEDIUM / HIGH / CRITICAL

    def to_markdown(self) -> str:
        lines = [
            "# System Audit Report\n",
            f"**Risk Level:** {self.risk_score}\n",
            "## Architecture\n", self.architecture_summary, "\n",
            "## Security Findings\n",
        ]
        if self.security_findings:
            for f in self.security_findings[:10]:
                lines.append(f"- [{f.get('severity','?')}] {f.get('file','?')}:{f.get('line','?')} — {f.get('description','?')}")
        else:
            lines.append("No security findings.")
        lines += [
            "\n## Dependencies\n", self.dependency_health, "\n",
            "## Test Coverage\n", self.test_coverage_summary, "\n",
            "## Full Report\n", self.audit_report,
        ]
        return "\n".join(lines)


class AuditWorkflow:
    """Multi-agent audit: architecture → security → deps → report."""

    async def run(
        self,
        repo_path: str,
        prompt: str,
        context: Any,
        arch_model: str,
        security_model: str,
        doc_model: str,
    ) -> AuditWorkflowResult:
        from app.agents.agent_factory import get_agent_factory
        from app.agents.base_agent import AgentTask

        start = time.monotonic()
        factory = get_agent_factory()

        # Step 1: Architecture overview
        arch_agent = factory.get_agent("architecture")
        arch_task = AgentTask(
            prompt=f"Provide a structural overview of this codebase for an audit: {prompt}",
            repo_path=repo_path,
            context=context,
            model=arch_model,
        )
        arch_result = await arch_agent.run(arch_task)

        # Step 2: Security scan
        security_agent = factory.get_agent("security")
        sec_task = AgentTask(
            prompt=f"Security audit: {prompt}",
            repo_path=repo_path,
            context=context,
            model=security_model,
        )
        sec_result = await security_agent.run(sec_task)
        findings = sec_result.details.get("findings", [])

        # Step 3: Dependency analysis (via tool directly)
        from app.tools.code_analysis.dependency_analyzer import DependencyAnalyzer
        dep_tool = DependencyAnalyzer()
        dep_result = await dep_tool.execute(path=repo_path)

        # Step 4: Generate report
        doc_agent = factory.get_agent("documentation")
        report_prompt = (
            f"Produce a comprehensive audit report based on:\n\n"
            f"ARCHITECTURE:\n{arch_result.response[:1000]}\n\n"
            f"SECURITY ({len(findings)} findings):\n{sec_result.response[:1000]}\n\n"
            f"DEPENDENCIES:\n{dep_result.output[:500]}\n\n"
            f"Original request: {prompt}"
        )
        doc_task = AgentTask(
            prompt=report_prompt,
            repo_path=repo_path,
            context=context,
            model=doc_model,
        )
        doc_result = await doc_agent.run(doc_task)

        # Risk scoring
        critical_count = sum(1 for f in findings if f.get("severity") == "CRITICAL")
        high_count = sum(1 for f in findings if f.get("severity") == "HIGH")
        risk_score = (
            "CRITICAL" if critical_count > 0
            else "HIGH" if high_count > 2
            else "MEDIUM" if high_count > 0
            else "LOW"
        )

        return AuditWorkflowResult(
            architecture_summary=arch_result.response,
            security_findings=findings,
            dependency_health=dep_result.output,
            test_coverage_summary="Run pytest --cov for coverage details",
            audit_report=doc_result.response,
            duration_ms=(time.monotonic() - start) * 1000,
            models_used=[arch_model, security_model, doc_model],
            risk_score=risk_score,
        )
