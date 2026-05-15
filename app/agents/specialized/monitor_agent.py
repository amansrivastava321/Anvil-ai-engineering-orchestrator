"""Proactive Monitoring Agent — watches repositories continuously.

This is the first step toward autonomous operation. The agent polls a repository
for git changes, runs the test suite after each change, and automatically triggers
the debug workflow when tests fail.

Design principles:
- Never raises — all errors are recorded in the status report
- Non-blocking: uses asyncio.sleep so the event loop stays free
- Observable: every decision is logged at INFO level
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base_agent import AgentTask, AgentResult, BaseAgent, BaseTool, ToolResult
from app.core.monitoring.logging import get_logger
from app.integrations.ollama.client import OllamaClient

logger = get_logger(__name__)

__all__ = ["MonitorAgent", "RepoStatus", "ChangeEvent"]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ChangeEvent:
    """Represents a detected change in a repository."""
    repo_path: str
    commit_hash: str
    changed_files: List[str]
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    branch: str = "unknown"

    def __str__(self) -> str:
        return (
            f"ChangeEvent({self.commit_hash[:8]}, "
            f"{len(self.changed_files)} files, "
            f"branch={self.branch})"
        )


@dataclass
class RepoStatus:
    """Current monitoring status for a single repository."""
    repo_path: str
    watching: bool = False
    last_commit: Optional[str] = None
    last_checked: Optional[datetime] = None
    last_test_run: Optional[datetime] = None
    test_passing: Optional[bool] = None
    test_summary: str = ""
    change_count: int = 0
    debug_triggers: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "watching": self.watching,
            "last_commit": self.last_commit,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_test_run": self.last_test_run.isoformat() if self.last_test_run else None,
            "test_passing": self.test_passing,
            "test_summary": self.test_summary,
            "change_count": self.change_count,
            "debug_triggers": self.debug_triggers,
            "recent_errors": self.errors[-5:],
        }


# ── MonitorAgent ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a Proactive Engineering Monitor — an autonomous agent that continuously watches code repositories and detects quality regressions before they reach production.

Your job is to:
1. Analyze test failures and determine their root cause
2. Identify which recent code changes caused the failures
3. Produce a concise, actionable diagnosis

When analyzing a test failure:
- Lead with the EXACT error message and failing test name
- Identify the specific code change most likely responsible
- State the fix in one clear sentence
- Note any related tests that may also be affected

Be precise, not verbose. Engineers reading your output are already stressed."""


class MonitorAgent(BaseAgent):
    """Proactive monitor: watches repos for changes and triggers analysis on failures.

    Usage::

        agent = MonitorAgent(ollama_client=client)
        await agent.watch_repository("/path/to/repo", poll_interval=60)
    """

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        super().__init__(ollama_client=ollama_client)
        self._repo_statuses: Dict[str, RepoStatus] = {}
        self._watch_tasks: Dict[str, asyncio.Task] = {}

    # ── BaseAgent abstract properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "monitor"

    @property
    def description(self) -> str:
        return "Proactive repository monitor: detects changes, runs tests, triggers debug on failure"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def tools(self) -> list[BaseTool]:
        return []  # monitor uses direct subprocess, not agent tools

    async def _execute(self, task: AgentTask) -> str:
        """Analyse a test failure and produce a diagnosis."""
        return await self._call_model(
            messages=self._build_messages(task),
            model=self._ollama.default_model,
            stream=False,
        )

    # ── Public monitoring API ──────────────────────────────────────────────────

    async def watch_repository(
        self,
        repo_path: str,
        poll_interval: int = 60,
        auto_debug: bool = True,
    ) -> None:
        """Continuously monitor a repository for git changes.

        Polls every ``poll_interval`` seconds. On change: runs tests. On
        test failure (and ``auto_debug=True``): triggers the debug workflow.

        This coroutine runs until cancelled.
        """
        path = Path(repo_path).resolve()
        if not (path / ".git").exists():
            logger.warning("Not a git repo — skipping", repo_path=str(path))
            return

        status = RepoStatus(repo_path=str(path), watching=True)
        self._repo_statuses[str(path)] = status

        logger.info(
            "Watching repository",
            repo_path=str(path),
            poll_interval=poll_interval,
            auto_debug=auto_debug,
        )

        # Bootstrap: record current HEAD so we don't fire on first poll
        status.last_commit = self._current_commit(path)
        status.last_checked = datetime.now(timezone.utc)

        while True:
            try:
                await asyncio.sleep(poll_interval)
                await self._poll_once(path, status, auto_debug)
            except asyncio.CancelledError:
                status.watching = False
                logger.info("Watch cancelled", repo_path=str(path))
                return
            except Exception as exc:
                status.errors.append(f"{datetime.now(timezone.utc).isoformat()}: {exc}")
                logger.error("Poll error", repo_path=str(path), error=str(exc))

    async def stop_watching(self, repo_path: str) -> bool:
        """Cancel the watch task for a specific repository."""
        key = str(Path(repo_path).resolve())
        task = self._watch_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        status = self._repo_statuses.get(key)
        if status:
            status.watching = False
        return False

    async def on_change_detected(
        self,
        repo_path: Path,
        changes: ChangeEvent,
    ) -> Dict[str, Any]:
        """Handle detected changes: run tests and analyse results.

        Returns a dict with keys: ``tests_passed``, ``summary``, ``triggered_debug``.
        """
        logger.info(
            "Change detected",
            repo_path=str(repo_path),
            commit=changes.commit_hash[:8],
            changed_files=len(changes.changed_files),
        )

        test_result = await self._run_tests(repo_path)
        status = self._repo_statuses.get(str(repo_path), RepoStatus(str(repo_path)))
        status.last_test_run = datetime.now(timezone.utc)
        status.test_passing = test_result["passed"]
        status.test_summary = test_result["summary"]
        status.change_count += 1

        return {
            "tests_passed": test_result["passed"],
            "summary": test_result["summary"],
            "triggered_debug": False,
        }

    async def on_test_failure(
        self,
        repo_path: Path,
        test_results: Dict[str, Any],
        changes: ChangeEvent,
    ) -> Dict[str, Any]:
        """Handle test failures: produce an AI diagnosis.

        Returns a dict with the AI analysis and which commit caused the failure.
        """
        status = self._repo_statuses.get(str(repo_path))
        if status:
            status.debug_triggers += 1

        logger.info(
            "Test failure detected — triggering analysis",
            repo_path=str(repo_path),
            failed=test_results.get("failed", "?"),
            commit=changes.commit_hash[:8],
        )

        prompt = (
            f"Repository: {repo_path}\n"
            f"Commit: {changes.commit_hash[:12]} (branch: {changes.branch})\n"
            f"Changed files: {', '.join(changes.changed_files[:10])}\n\n"
            f"Test output:\n{test_results.get('output', '')[-3000:]}\n\n"
            f"Diagnose the failure and identify which change caused it."
        )

        task = AgentTask(prompt=prompt, repo_path=str(repo_path))
        result = await self.run(task)

        return {
            "diagnosis": result.response,
            "commit": changes.commit_hash,
            "changed_files": changes.changed_files,
            "test_summary": test_results.get("summary", ""),
            "success": result.status.value == "completed",
        }

    async def generate_status_report(self) -> Dict[str, Any]:
        """Generate current monitoring status for all watched repositories."""
        return {
            "watching": len([s for s in self._repo_statuses.values() if s.watching]),
            "total_repos": len(self._repo_statuses),
            "repos": {
                path: status.to_dict()
                for path, status in self._repo_statuses.items()
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _poll_once(
        self,
        repo_path: Path,
        status: RepoStatus,
        auto_debug: bool,
    ) -> None:
        """One polling cycle: check for new commits, run tests on change."""
        current = self._current_commit(repo_path)
        status.last_checked = datetime.now(timezone.utc)

        if current == status.last_commit:
            return  # no change

        # New commit detected
        prev = status.last_commit or "HEAD~1"
        changed_files = self._diff_files(repo_path, prev, current)
        branch = self._current_branch(repo_path)

        event = ChangeEvent(
            repo_path=str(repo_path),
            commit_hash=current,
            changed_files=changed_files,
            branch=branch,
        )
        status.last_commit = current

        result = await self.on_change_detected(repo_path, event)

        if not result["tests_passed"] and auto_debug:
            test_output = await self._run_tests(repo_path)
            await self.on_test_failure(repo_path, test_output, event)

    async def _run_tests(self, repo_path: Path) -> Dict[str, Any]:
        """Run pytest in the repository and return structured results."""
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", "--tb=short", "-q", "--no-cov",
                cwd=str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "passed": False,
                    "summary": "Test run timed out after 120s",
                    "output": "TIMEOUT",
                    "duration_ms": (time.monotonic() - start) * 1000,
                }

            output = stdout.decode(errors="replace")
            passed = proc.returncode == 0

            # Parse summary line (e.g. "5 passed, 1 failed in 1.23s")
            summary = ""
            for line in reversed(output.splitlines()):
                if "passed" in line or "failed" in line or "error" in line:
                    summary = line.strip()
                    break

            return {
                "passed": passed,
                "summary": summary or ("All tests passed" if passed else "Tests failed"),
                "output": output[-4000:],
                "duration_ms": (time.monotonic() - start) * 1000,
                "failed": 0 if passed else 1,
            }
        except Exception as exc:
            return {
                "passed": False,
                "summary": f"Test runner error: {exc}",
                "output": str(exc),
                "duration_ms": (time.monotonic() - start) * 1000,
            }

    @staticmethod
    def _current_commit(repo_path: Path) -> str:
        """Return the current HEAD commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _current_branch(repo_path: Path) -> str:
        """Return the current branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _diff_files(repo_path: Path, from_ref: str, to_ref: str) -> List[str]:
        """Return list of files changed between two refs."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", from_ref, to_ref],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [f for f in result.stdout.strip().splitlines() if f]
        except Exception:
            pass
        return []
