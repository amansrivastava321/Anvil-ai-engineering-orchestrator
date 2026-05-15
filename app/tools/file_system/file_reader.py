"""FileReader — read repository files safely."""

import time
from pathlib import Path
from typing import Any

import aiofiles

from app.agents.base_agent import BaseTool, ToolResult
from app.utils.validators import PathValidator

__all__ = ["FileReader", "ToolResult"]


class FileReader(BaseTool):
    """Tool: read a file from the repository by relative path.

    Prevents path traversal attacks via PathValidator.
    """

    def __init__(self, repo_path: str) -> None:
        self._repo = Path(repo_path)
        # Register the repo root so PathValidator permits reads within it.
        resolved_repo = self._repo.resolve()
        if resolved_repo not in PathValidator.ALLOWED_BASE_DIRS:
            PathValidator.set_allowed_base_dirs(
                PathValidator.ALLOWED_BASE_DIRS + [resolved_repo]
            )

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the full contents of a file. "
            "Input: file_path (string, relative to the repository root)."
        )

    async def execute(self, file_path: str = "", **_: Any) -> ToolResult:
        start = time.monotonic()
        try:
            validated = PathValidator.validate_path(
                str(self._repo / file_path),
                must_exist=True,
                must_be_file=True,
            )
            async with aiofiles.open(validated) as fh:
                content = await fh.read()
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=content,
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
