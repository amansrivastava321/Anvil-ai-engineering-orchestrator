"""3-agent refactoring pipeline: Analyze → Refactor → Review."""

import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RefactorWorkflowResult:
    analysis: str
    refactored_code: str
    review: str
    steps_taken: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_markdown(self) -> str:
        return (
            f"## Architecture Analysis\n\n{self.analysis}"
            f"\n\n## Refactored Code\n\n{self.refactored_code}"
            f"\n\n## Review\n\n{self.review}"
        )


class RefactorWorkflow:
    """3-agent refactoring pipeline.

    1. ArchitectureAgent — identify what to refactor and why
    2. CodeAgent — perform the refactoring
    3. CodeAgent — review the refactored output for regressions
    """

    def __init__(self, factory: AgentFactory | None = None) -> None:
        self._factory = factory or get_agent_factory()

    async def run(
        self,
        repo_path: str,
        prompt: str,
        context: Any | None,
        arch_model: str,
        code_model: str,
    ) -> RefactorWorkflowResult:
        start = time.monotonic()
        steps: list[str] = []

        logger.info("Refactor workflow starting", repo_path=repo_path)

        arch_agent = self._factory.get_agent("architecture")
        code_agent = self._factory.get_agent("code")

        analysis_result = await arch_agent.run(AgentTask(
            prompt=(
                f"Analyze the following code for refactoring opportunities.\n\n{prompt}\n\n"
                "Identify: what should change, why, and what the risk is."
            ),
            repo_path=repo_path,
            context=context,
            model=arch_model,
            temperature=0.2,
        ))
        steps.append("architecture_analysis")

        refactor_result = await code_agent.run(AgentTask(
            prompt=(
                f"Refactor this code based on the analysis below.\n\n"
                f"Analysis:\n{analysis_result.response}\n\n"
                f"Code:\n{prompt}\n\n"
                "Output the complete refactored implementation. Preserve all public contracts."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("refactoring")

        review_result = await code_agent.run(AgentTask(
            prompt=(
                f"Review this refactored code.\n\nOriginal:\n{prompt}\n\n"
                f"Refactored:\n{refactor_result.response}\n\n"
                "Identify any bugs, broken contracts, or security issues introduced."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("code_review")

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("Refactor workflow completed", duration_ms=round(duration_ms, 2))

        return RefactorWorkflowResult(
            analysis=analysis_result.response,
            refactored_code=refactor_result.response,
            review=review_result.response,
            steps_taken=steps,
            duration_ms=duration_ms,
        )
