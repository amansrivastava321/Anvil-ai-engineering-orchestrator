"""File search tool — search filenames and content, skip .git/node_modules/.venv."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agents.base_agent import BaseTool, ToolResult

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", "dist", "build", ".tox"}


class FileSearch(BaseTool):
    """Search files by name pattern or content across a repository."""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return (
            "Search for files by name pattern or content. "
            "Args: repo_path (str), query (str), search_content (bool=True), "
            "file_pattern (str='*.py'), max_results (int=20)"
        )

    async def execute(  # type: ignore[override]
        self,
        repo_path: str = "",
        query: str = "",
        search_content: bool = True,
        file_pattern: str = "*.py",
        max_results: int = 20,
        **_: Any,
    ) -> ToolResult:
        if not repo_path or not query:
            return ToolResult(tool_name=self.name, success=False, output="", error="repo_path and query required")

        root = Path(repo_path)
        if not root.is_dir():
            return ToolResult(tool_name=self.name, success=False, output="", error=f"Directory not found: {repo_path}")

        results: list[dict[str, Any]] = []
        query_lower = query.lower()
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for path in root.rglob(file_pattern):
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if len(results) >= max_results:
                break

            rel = str(path.relative_to(root))

            # Filename match
            if query_lower in path.name.lower():
                results.append({"file": rel, "match_type": "filename", "line": None, "snippet": None})
                continue

            # Content match
            if search_content:
                try:
                    text = path.read_text(errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pattern.search(line):
                            results.append({
                                "file": rel,
                                "match_type": "content",
                                "line": i,
                                "snippet": line.strip()[:120],
                            })
                            if len(results) >= max_results:
                                break
                except Exception:
                    pass

        if not results:
            return ToolResult(tool_name=self.name, success=True, output=f"No results for '{query}'")

        lines = [f"Found {len(results)} result(s) for '{query}':"]
        for r in results:
            loc = f":{r['line']}" if r["line"] else ""
            snippet = f" — {r['snippet']}" if r["snippet"] else ""
            lines.append(f"  {r['file']}{loc}{snippet}")

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
        )
