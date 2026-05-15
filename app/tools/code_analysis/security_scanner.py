"""Security scanner — detect secrets, unsafe patterns, hardcoded credentials."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agents.base_agent import BaseTool, ToolResult

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__"}

# Pattern → severity
PATTERNS: list[tuple[str, str, str]] = [
    (r'(?i)(password|passwd|secret|api_key|apikey|token|auth_token)\s*=\s*["\'][^"\']{4,}["\']', "HIGH", "Hardcoded credential"),
    (r'(?i)aws_access_key_id\s*=\s*["\'][A-Z0-9]{20}["\']', "CRITICAL", "AWS access key"),
    (r'(?i)aws_secret_access_key\s*=\s*["\'][^"\']{30,}["\']', "CRITICAL", "AWS secret key"),
    (r'\beval\s*\(', "HIGH", "eval() usage"),
    (r'\bexec\s*\(', "MEDIUM", "exec() usage"),
    (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', "HIGH", "shell=True in subprocess"),
    (r'pickle\.loads?\s*\(', "MEDIUM", "pickle deserialization"),
    (r'__import__\s*\(', "MEDIUM", "Dynamic import"),
    (r'(?i)(private_key|privatekey|-----BEGIN\s+(?:RSA\s+)?PRIVATE)', "CRITICAL", "Private key material"),
    (r'(?i)(sk-[a-zA-Z0-9]{20,})', "HIGH", "Potential API key"),
]

COMPILED = [(re.compile(p), sev, desc) for p, sev, desc in PATTERNS]


class SecurityScanner(BaseTool):
    """Scan Python files for security vulnerabilities."""

    @property
    def name(self) -> str:
        return "scan_security"

    @property
    def description(self) -> str:
        return (
            "Scan a file or directory for security vulnerabilities. "
            "Args: path (str), recursive (bool=True)"
        )

    async def execute(  # type: ignore[override]
        self,
        path: str = "",
        recursive: bool = True,
        **_: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(tool_name=self.name, success=False, output="", error="path required")

        target = Path(path)
        if not target.exists():
            return ToolResult(tool_name=self.name, success=False, output="", error=f"Path not found: {path}")

        files = (
            [f for f in target.rglob("*.py") if not any(s in f.parts for s in SKIP_DIRS)]
            if target.is_dir() and recursive
            else [target] if target.is_file()
            else []
        )

        findings: list[dict[str, Any]] = []
        for f in files:
            try:
                text = f.read_text(errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    for compiled, severity, desc in COMPILED:
                        if compiled.search(line):
                            findings.append({
                                "file": str(f),
                                "line": i,
                                "severity": severity,
                                "description": desc,
                                "snippet": line.strip()[:100],
                            })
            except Exception:
                pass

        if not findings:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="No security issues found.",
            )

        lines = [f"Security scan: {len(findings)} finding(s) in {len(files)} file(s):\n"]
        for f in findings:
            lines.append(f"  [{f['severity']}] {f['file']}:{f['line']} — {f['description']}")
            lines.append(f"    {f['snippet']}")

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
        )
