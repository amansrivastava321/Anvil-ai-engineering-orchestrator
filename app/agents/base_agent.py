"""Base agent abstraction for all specialized engineering agents.

Agents are NOT Ollama wrappers. They are reasoning entities with:
- A focused engineering domain (code, architecture, testing, docs)
- A structured system prompt expressing expertise and discipline
- Observable execution via structured logs
- Tool support for file access and test running
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from app.core.monitoring.logging import get_logger
from app.integrations.ollama.client import OllamaClient, get_default_client

__all__ = [
    "AgentStatus",
    "ToolResult",
    "AgentStep",
    "AgentResult",
    "AgentTask",
    "BaseTool",
    "BaseAgent",
]


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolResult:
    """Result from a single tool execution."""

    tool_name: str
    success: bool
    output: str
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class AgentStep:
    """One reasoning step recorded during agent execution."""

    thought: str
    action: str | None = None
    tool_used: str | None = None
    tool_input: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    """Complete result from an agent execution."""

    agent_name: str
    execution_id: str
    status: AgentStatus
    response: str
    steps: list[AgentStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tokens_approx: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    confidence: float = 1.0
    next_actions: list[str] = field(default_factory=list)


class AgentTask(BaseModel):
    """Task submitted to an agent for execution."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    repo_path: str
    # Typed loosely to avoid circular import with AssembledContext
    context: Any | None = None
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.2
    max_tokens: int = 4096
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class BaseTool(ABC):
    """Abstract base for tools that agents can invoke."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used in agent reasoning and logging."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description injected into the agent's system prompt."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool. Must not raise — return ToolResult with success=False on error."""


class BaseAgent(ABC):
    """Abstract base for all specialized engineering agents.

    Subclasses must implement:
    - name (str property)
    - description (str property)
    - system_prompt (str property)
    - _execute(task: AgentTask) -> str

    Optionally override:
    - tools (list[BaseTool] property) to give the agent file/test access
    """

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        self.ollama = ollama_client or get_default_client()
        self._logger = get_logger(f"agent.{self.name}")

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier — used in logs and metrics."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of this agent's specialization."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Core system prompt. Defines the agent's expertise and discipline."""

    @property
    def tools(self) -> list[BaseTool]:
        """Tools available to this agent. Override to add domain-specific tools."""
        return []

    @abstractmethod
    async def _execute(self, task: AgentTask) -> str:
        """Domain-specific execution logic. Called by run()."""

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentResult:
        """Execute task synchronously and return a complete AgentResult."""
        execution_id = str(uuid.uuid4())
        start = time.monotonic()

        self._logger.info(
            "Agent starting",
            agent=self.name,
            task_id=task.task_id,
            execution_id=execution_id,
            model=task.model,
        )

        try:
            response = await self._execute(task)
            duration_ms = (time.monotonic() - start) * 1000
            tokens_approx = self._approx_tokens(response)

            self._logger.info(
                "Agent completed",
                agent=self.name,
                execution_id=execution_id,
                duration_ms=round(duration_ms, 2),
                tokens_approx=tokens_approx,
            )

            return AgentResult(
                agent_name=self.name,
                execution_id=execution_id,
                status=AgentStatus.COMPLETED,
                response=response,
                tokens_approx=tokens_approx,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._logger.error(
                "Agent failed",
                agent=self.name,
                execution_id=execution_id,
                error=str(exc),
                exc_info=True,
            )
            return AgentResult(
                agent_name=self.name,
                execution_id=execution_id,
                status=AgentStatus.FAILED,
                response="",
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def run_streaming(self, task: AgentTask) -> AsyncIterator[str]:
        """Stream response tokens as they are generated."""
        execution_id = str(uuid.uuid4())
        start = time.monotonic()
        self._logger.info(
            "Agent streaming",
            agent=self.name,
            task_id=task.task_id,
            execution_id=execution_id,
        )
        messages = self._build_messages(task)
        try:
            async for token in self.ollama.chat(
                model=task.model,
                messages=messages,
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                stream=True,
            ):
                yield token
            self._logger.info(
                "Agent stream completed",
                agent=self.name,
                execution_id=execution_id,
                duration_ms=round((time.monotonic() - start) * 1000, 2),
            )
        except Exception as exc:
            self._logger.error(
                "Agent stream failed",
                agent=self.name,
                execution_id=execution_id,
                error=str(exc),
                exc_info=True,
            )
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
        self, task: AgentTask, focus: str | None = None
    ) -> list[dict[str, str]]:
        """Assemble the message list passed to the model."""
        system = self._full_system_prompt()
        if focus:
            system += f"\n\nFOCUS: {focus}"

        if task.context is not None and hasattr(task.context, "user_prompt"):
            user_content = f"{task.context.user_prompt}\n\n---\n\n{task.prompt}"
        else:
            user_content = task.prompt

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _full_system_prompt(self) -> str:
        """System prompt with tool descriptions appended if tools are registered."""
        base = self.system_prompt
        if not self.tools:
            return base
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools
        )
        return f"{base}\n\nAvailable tools:\n{tool_descriptions}"

    async def _call_model(
        self,
        task: AgentTask,
        focus: str | None = None,
    ) -> str:
        """Execute a single synchronous model call."""
        messages = self._build_messages(task, focus=focus)
        return await self.ollama.chat(  # type: ignore[return-value]
            model=task.model,
            messages=messages,
            temperature=task.temperature,
            max_tokens=task.max_tokens,
            stream=False,
        )

    def _approx_tokens(self, text: str) -> int:
        return len(text) // 4
