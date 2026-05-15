"""TestRunner — run pytest in the repository and capture output."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from app.agents.base_agent import BaseTool, ToolResult

__all__ = ["TestRunner"]

_MAX_OUTPUT_CHARS = 8_000
_TIMEOUT_SECONDS = 120.0


class TestRunner(BaseTool):
    """Tool: run pytest tests and return the captured output.

    Never raises — always returns ToolResult.
    Output is capped at 8000 characters to stay within model context limits.
    """

    def __init__(self, repo_path: str) -> None:
        self._repo = Path(repo_path)

    @property
    def name(self) -> str:
        return "run_tests"

    @property
    def description(self) -> str:
        return (
            "Run pytest tests in the repository. "
            "Inputs: test_path (optional, relative path to test file or directory), "
            "extra_args (optional string of additional pytest flags). "
            "Returns test output and pass/fail status."
        )

    async def execute(
        self,
        test_path: str | None = None,
        extra_args: str | None = None,
        **_: Any,
    ) -> ToolResult:
        start = time.monotonic()
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "--no-header", "--no-cov"]

        if test_path:
            cmd.append(test_path)
        if extra_args:
            cmd.extend(extra_args.split())

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_TIMEOUT_SECONDS
            )
            output = stdout.decode(errors="replace")[:_MAX_OUTPUT_CHARS]
            success = proc.returncode == 0
            return ToolResult(
                tool_name=self.name,
                success=success,
                output=output,
                error=None if success else f"Tests failed (exit code {proc.returncode})",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Test run timed out after {_TIMEOUT_SECONDS}s",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )
