"""2-agent testing pipeline: Strategy → Generate tests."""

import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestingWorkflowResult:
    test_strategy: str
    generated_tests: str
    steps_taken: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_markdown(self) -> str:
        return (
            f"## Test Strategy\n\n{self.test_strategy}"
            f"\n\n## Generated Tests\n\n{self.generated_tests}"
        )


class TestingWorkflow:
    """2-agent testing pipeline.

    1. TestingAgent — design the test strategy
    2. CodeAgent — generate complete, runnable pytest tests from the strategy
    """

    def __init__(self, factory: AgentFactory | None = None) -> None:
        self._factory = factory or get_agent_factory()

    async def run(
        self,
        prompt: str,
        repo_path: str,
        context: Any | None,
        testing_model: str,
        code_model: str,
    ) -> TestingWorkflowResult:
        start = time.monotonic()
        steps: list[str] = []

        logger.info("Testing workflow starting", repo_path=repo_path)

        testing_agent = self._factory.get_agent("testing")
        code_agent = self._factory.get_agent("code")

        strategy_result = await testing_agent.run(AgentTask(
            prompt=(
                f"Design a test strategy for the following:\n\n{prompt}\n\n"
                "Specify: which behaviors to test, what level (unit/integration/e2e), "
                "what to mock and why, key edge cases, and required fixtures."
            ),
            repo_path=repo_path,
            context=context,
            model=testing_model,
            temperature=0.2,
        ))
        steps.append("test_strategy")

        test_result = await code_agent.run(AgentTask(
            prompt=(
                f"Generate complete pytest tests based on this strategy.\n\n"
                f"Strategy:\n{strategy_result.response}\n\n"
                f"Code to test:\n{prompt}\n\n"
                "Output complete, runnable .py files with all imports."
            ),
            repo_path=repo_path,
            context=context,
            model=code_model,
            temperature=0.1,
        ))
        steps.append("test_generation")

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("Testing workflow completed", duration_ms=round(duration_ms, 2))

        return TestingWorkflowResult(
            test_strategy=strategy_result.response,
            generated_tests=test_result.response,
            steps_taken=steps,
            duration_ms=duration_ms,
        )
