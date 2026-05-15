"""File chunker — split large source files by function/class boundaries.

Never splits a function in two. Supports Python precisely (via ast) and
other languages via regex heuristics.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CodeChunk:
    file_path: str
    function_name: str       # "module" for top-level code, else the def/class name
    start_line: int
    end_line: int
    content: str


class FileChunker:
    """Split files into logical units without breaking function boundaries."""

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Split content into one chunk per function/class (never mid-function).

        Falls back to fixed-size line chunks if language is not Python.
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            return self._chunk_python(file_path, content)
        return self._chunk_by_regex(file_path, content)

    def extract_skeleton(self, file_path: str, content: str) -> str:
        """Return only signatures and docstrings — skip function bodies.

        Turns a 500-line file into ~40 lines. Typical reduction: 90%.
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            return self._python_skeleton(file_path, content)
        return self._regex_skeleton(content)

    def extract_function(self, content: str, function_name: str) -> str:
        """Extract ONE named function/method with its full body."""
        # Try Python AST first
        try:
            tree = ast.parse(content)
            lines = content.splitlines(keepends=True)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == function_name:
                        end = self._ast_end_line(node, len(lines))
                        return "".join(lines[node.lineno - 1 : end])
        except SyntaxError:
            pass

        # Regex fallback (works for JS/TS/Go/etc.)
        pattern = rf"((?:^[ \t]*(?:async\s+)?(?:def|function|func|fn|sub)\s+{re.escape(function_name)}\b.*?(?=\n(?:[ \t]*(?:async\s+)?(?:def|function|func|fn|sub)\s+|\Z))))"
        m = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        return m.group(1) if m else f"# Function '{function_name}' not found\n"

    def extract_with_dependencies(self, content: str, function_name: str) -> str:
        """Extract function_name plus all functions it calls within the same file."""
        try:
            tree = ast.parse(content)
            lines = content.splitlines(keepends=True)

            # Find the target function node
            target_node = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        target_node = node
                        break

            if target_node is None:
                return self.extract_function(content, function_name)

            # Collect names called inside target
            called_names = {
                node.func.id
                for node in ast.walk(target_node)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }

            # Gather the target + all called functions defined in the same file
            parts = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name or node.name in called_names:
                        end = self._ast_end_line(node, len(lines))
                        parts.append((node.lineno, "".join(lines[node.lineno - 1 : end])))

            parts.sort(key=lambda x: x[0])
            return "\n\n".join(code for _, code in parts) or self.extract_function(content, function_name)
        except SyntaxError:
            return self.extract_function(content, function_name)

    # ── Python-specific ───────────────────────────────────────────────────────

    def _chunk_python(self, file_path: str, content: str) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        lines = content.splitlines(keepends=True)
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._chunk_by_regex(file_path, content)

        top_level = [
            n for n in ast.iter_child_nodes(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not top_level:
            # File has no top-level defs — return as one chunk
            return [CodeChunk(
                file_path=file_path, function_name="module",
                start_line=1, end_line=len(lines),
                content=content,
            )]

        prev_end = 0
        for node in top_level:
            start = node.lineno - 1
            end = self._ast_end_line(node, len(lines))

            # Any top-level code before this def → its own "module" chunk
            if start > prev_end:
                preamble = "".join(lines[prev_end:start])
                if preamble.strip():
                    chunks.append(CodeChunk(
                        file_path=file_path, function_name="module",
                        start_line=prev_end + 1, end_line=start,
                        content=preamble,
                    ))
            chunks.append(CodeChunk(
                file_path=file_path, function_name=node.name,
                start_line=node.lineno, end_line=end,
                content="".join(lines[start:end]),
            ))
            prev_end = end

        # Trailing code
        if prev_end < len(lines):
            tail = "".join(lines[prev_end:])
            if tail.strip():
                chunks.append(CodeChunk(
                    file_path=file_path, function_name="module",
                    start_line=prev_end + 1, end_line=len(lines),
                    content=tail,
                ))
        return chunks

    def _python_skeleton(self, file_path: str, content: str) -> str:
        """Signatures + first docstring line only."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._regex_skeleton(content)

        lines = content.splitlines()
        out: List[str] = []

        # Module-level imports
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                out.append(lines[node.lineno - 1])

        out.append("")

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            sig = lines[node.lineno - 1].rstrip()
            # Capture multi-line signature (args continuing onto next lines)
            i = node.lineno - 1
            while not sig.rstrip().endswith(":") and i < len(lines) - 1:
                i += 1
                sig += "\n" + lines[i].rstrip()
            out.append(sig)

            # First docstring line only
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc = node.body[0].value.value.strip().splitlines()[0]
                indent = "    " if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "    "
                out.append(f'{indent}"""' + doc + '"""')

            out.append("")  # blank line between defs

        return "\n".join(out)

    # ── Regex-based fallback ──────────────────────────────────────────────────

    _DEF_RE = re.compile(
        r"^([ \t]*(?:export\s+)?(?:async\s+)?(?:def|function|func|fn|sub|proc)\s+\w+[^{;]*[{:]?)",
        re.MULTILINE,
    )

    def _chunk_by_regex(self, file_path: str, content: str) -> List[CodeChunk]:
        lines = content.splitlines(keepends=True)
        boundaries = [m.start() for m in self._DEF_RE.finditer(content)]
        if not boundaries:
            return [CodeChunk(
                file_path=file_path, function_name="module",
                start_line=1, end_line=len(lines),
                content=content,
            )]
        chunks: List[CodeChunk] = []
        char_to_line = self._char_to_line_map(content)
        for idx, start_char in enumerate(boundaries):
            end_char = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(content)
            chunk_content = content[start_char:end_char]
            name_m = re.search(r"(?:def|function|func|fn|sub|proc)\s+(\w+)", chunk_content)
            name = name_m.group(1) if name_m else "unknown"
            start_line = char_to_line[start_char]
            end_line = char_to_line[min(end_char - 1, len(content) - 1)]
            chunks.append(CodeChunk(
                file_path=file_path, function_name=name,
                start_line=start_line, end_line=end_line,
                content=chunk_content,
            ))
        return chunks

    def _regex_skeleton(self, content: str) -> str:
        """Return only function signature lines for non-Python files."""
        out = []
        for m in self._DEF_RE.finditer(content):
            out.append(m.group(1).rstrip())
        return "\n".join(out)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ast_end_line(node: ast.AST, total_lines: int) -> int:
        """Return the exclusive end line index (0-based) for a node."""
        if hasattr(node, "end_lineno") and node.end_lineno:
            return node.end_lineno  # already 1-based exclusive
        return total_lines  # fallback: rest of file

    @staticmethod
    def _char_to_line_map(content: str) -> List[int]:
        """Map character offset → 1-based line number."""
        mapping = [0] * (len(content) + 1)
        line = 1
        for i, ch in enumerate(content):
            mapping[i] = line
            if ch == "\n":
                line += 1
        mapping[len(content)] = line
        return mapping
