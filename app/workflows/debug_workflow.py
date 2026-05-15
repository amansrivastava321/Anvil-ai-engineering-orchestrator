"""3-agent debug pipeline: Architecture context → Root cause → Solution."""

import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DebugWorkflowResult:
    root_cause_analysis: str
    solution: str
    steps_taken: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    models_used: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        return (
            f"## Root Cause Analysis\n\n{self.root_cause_analysis}"
            f"\n\n## Solution\n\n{self.solution}"
        )


class DebugWorkflow:
    """3-agent debug pipeline.

    1. ArchitectureAgent — map structure and data flow around the issue
    2. CodeAgent — identify root cause from the architecture context
    3. CodeAgent — generate the fix from the root cause
    """

    def __init__(self, factory: AgentFactory | None = None) -> None:
        self._factory = factory or get_agent_factory()

    async def run(
        self,
        prompt: str,
        repo_path: str,
        context: Any | None,
        debug_model: str,
        code_model: str,
    ) -> DebugWorkflowResult:
        start = time.monotonic()
        steps: list[str] = []

        logger.info("Debug workflow starting", repo_path=repo_path)

        arch_agent = self._factory.get_agent("architecture")
        code_agent = self._factory.get_agent("code")

        arch_result = await arch_agent.run(AgentTask(
            prompt=(
                f"Map the architecture and data flow relevant to this issue:\n\n{prompt}\n\n"
                "Which modules, functions, and data paths are involved?"
            ),
            repo_path=repo_path,
            context=context,
            model=debug_model,
            temperature=0.1,
        ))
        steps.append("architecture_analysis")

        rca_result = await code_agent.run(AgentTask(
            prompt=(
                f"Issue:\n{prompt}\n\n"
                f"Architecture context:\n{arch_result.response}\n\n"
                "Identify the root cause. Be specific: which file, function, or interaction "
                "is the origin of the failure, and exactly why does it fail?"
            ),
            repo_path=repo_path,
            context=context,
            model=debug_model,
            temperature=0.1,
        ))
        steps.append("root_cause_analysis")

        solution_result = await code_agent.run(AgentTask(
            prompt=(
                f"Root cause:\n{rca_result.response}\n\n"
                "Generate the fix. Show the complete modified function or class, "
                "not just a diff. Include any new tests needed to prevent regression."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("solution_generation")

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("Debug workflow completed", duration_ms=round(duration_ms, 2))

        return DebugWorkflowResult(
            root_cause_analysis=rca_result.response,
            solution=solution_result.response,
            steps_taken=steps,
            duration_ms=duration_ms,
            models_used=list({debug_model, code_model}),
        )
