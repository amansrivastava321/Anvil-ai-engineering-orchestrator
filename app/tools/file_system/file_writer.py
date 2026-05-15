"""FileWriter — write files into the repository."""

import time
from pathlib import Path
from typing import Any

import aiofiles

from app.agents.base_agent import BaseTool, ToolResult
from app.utils.validators import PathValidator

__all__ = ["FileWriter"]


class FileWriter(BaseTool):
    """Tool: write content to a file in the repository.

    Creates the file if it does not exist. Overwrites if it does.
    Parent directory must already exist.
    """

    def __init__(self, repo_path: str) -> None:
        self._repo = Path(repo_path)
        # Register the repo root so PathValidator permits writes within it.
        resolved_repo = self._repo.resolve()
        if resolved_repo not in PathValidator.ALLOWED_BASE_DIRS:
            PathValidator.set_allowed_base_dirs(
                PathValidator.ALLOWED_BASE_DIRS + [resolved_repo]
            )

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. "
            "Inputs: file_path (relative to repo root), content (string). "
            "Creates the file if it does not exist. Overwrites if it does. "
            "Parent directory must already exist."
        )

    async def execute(self, file_path: str = "", content: str = "", **_: Any) -> ToolResult:
        start = time.monotonic()
        try:
            full_path = self._repo / file_path
            PathValidator.validate_path(
                str(full_path.parent),
                must_exist=True,
                must_be_dir=True,
            )
            async with aiofiles.open(full_path, "w") as fh:
                await fh.write(content)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"Wrote {len(content.encode())} bytes to {file_path}",
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
