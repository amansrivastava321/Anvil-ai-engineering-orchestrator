"""Dependency analyzer — build import graph for Python projects."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from app.agents.base_agent import BaseTool, ToolResult

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__"}


def _extract_imports(source: str) -> list[str]:
    """Extract top-level module names from Python source."""
    imports: list[str] = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])
    except SyntaxError:
        # Fallback: regex
        for m in re.finditer(r'^(?:import|from)\s+([\w.]+)', source, re.MULTILINE):
            imports.append(m.group(1).split(".")[0])
    return list(set(imports))


class DependencyAnalyzer(BaseTool):
    """Analyze import dependencies across a Python project."""

    @property
    def name(self) -> str:
        return "analyze_dependencies"

    @property
    def description(self) -> str:
        return (
            "Analyze import dependencies for a Python file or project directory. "
            "Args: path (str), include_stdlib (bool=False)"
        )

    async def execute(  # type: ignore[override]
        self,
        path: str = "",
        include_stdlib: bool = False,
        **_: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(tool_name=self.name, success=False, output="", error="path required")

        target = Path(path)
        if not target.exists():
            return ToolResult(tool_name=self.name, success=False, output="", error=f"Path not found: {path}")

        stdlib = _get_stdlib_names()
        graph: dict[str, list[str]] = {}

        files = (
            [f for f in target.rglob("*.py") if not any(s in f.parts for s in SKIP_DIRS)]
            if target.is_dir()
            else [target]
        )

        for f in files:
            try:
                source = f.read_text(errors="ignore")
                imports = _extract_imports(source)
                if not include_stdlib:
                    imports = [i for i in imports if i not in stdlib]
                graph[str(f)] = sorted(imports)
            except Exception:
                pass

        # Summary: unique external deps
        all_deps: set[str] = set()
        for deps in graph.values():
            all_deps.update(deps)

        lines = [f"Dependency analysis: {len(files)} file(s), {len(all_deps)} unique external imports\n"]
        lines.append("External dependencies:")
        for dep in sorted(all_deps):
            lines.append(f"  {dep}")

        if len(files) <= 10:
            lines.append("\nPer-file imports:")
            for file_path, deps in sorted(graph.items()):
                if deps:
                    lines.append(f"  {file_path}: {', '.join(deps)}")

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
        )


def _get_stdlib_names() -> set[str]:
    """Return a set of known stdlib top-level module names."""
    import sys
    return set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {
        "abc", "ast", "asyncio", "builtins", "collections", "contextlib",
        "copy", "dataclasses", "datetime", "enum", "functools", "hashlib",
        "inspect", "io", "itertools", "json", "logging", "math", "os",
        "pathlib", "pickle", "platform", "queue", "re", "shutil", "signal",
        "socket", "string", "subprocess", "sys", "tempfile", "threading",
        "time", "traceback", "typing", "unittest", "urllib", "uuid", "warnings",
    }
