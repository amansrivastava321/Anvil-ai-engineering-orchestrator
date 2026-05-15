"""ProactiveService — manages background repository monitoring.

Wraps MonitorAgent with lifecycle management (start, stop, status).
Designed to run as a singleton background service tied to the FastAPI
application lifespan.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from app.agents.specialized.monitor_agent import MonitorAgent
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ProactiveService", "get_proactive_service"]


class ProactiveService:
    """Background service that monitors repositories for changes.

    Usage::

        service = ProactiveService()
        await service.start_watching(["/path/to/repo"])
        ...
        await service.stop_watching()
    """

    def __init__(self) -> None:
        self._agent: Optional[MonitorAgent] = None
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start_watching(
        self,
        repos: List[str],
        poll_interval: int = 60,
        auto_debug: bool = True,
    ) -> Dict[str, Any]:
        """Start monitoring a list of repository paths.

        Already-watched repositories are skipped.  Returns a summary of
        what was started vs. skipped.
        """
        if self._agent is None:
            self._agent = MonitorAgent()

        self._running = True
        started: List[str] = []
        skipped: List[str] = []

        for repo in repos:
            if repo in self._tasks and not self._tasks[repo].done():
                skipped.append(repo)
                continue

            task = asyncio.create_task(
                self._agent.watch_repository(
                    repo,
                    poll_interval=poll_interval,
                    auto_debug=auto_debug,
                ),
                name=f"monitor:{repo}",
            )
            self._tasks[repo] = task
            started.append(repo)
            logger.info("Started watching repository", repo=repo)

        return {"started": started, "skipped": skipped, "total_watching": len(self._tasks)}

    async def stop_watching(self, repos: Optional[List[str]] = None) -> Dict[str, Any]:
        """Stop monitoring.  Pass ``repos`` to stop specific paths or
        ``None`` to stop all.
        """
        targets = repos if repos is not None else list(self._tasks.keys())
        stopped: List[str] = []
        not_found: List[str] = []

        for repo in targets:
            task = self._tasks.pop(repo, None)
            if task is None or task.done():
                not_found.append(repo)
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            stopped.append(repo)
            logger.info("Stopped watching repository", repo=repo)

        if not self._tasks:
            self._running = False

        return {"stopped": stopped, "not_found": not_found, "still_watching": len(self._tasks)}

    async def get_status(self) -> Dict[str, Any]:
        """Return current monitoring status for all watched repositories."""
        if self._agent is None:
            return {
                "running": False,
                "watching": 0,
                "total_repos": 0,
                "repos": {},
            }

        report = await self._agent.generate_status_report()
        report["running"] = self._running
        return report


# ── Singleton ──────────────────────────────────────────────────────────────────

_proactive_service: Optional[ProactiveService] = None


def get_proactive_service() -> ProactiveService:
    """Return the application-wide ProactiveService singleton."""
    global _proactive_service
    if _proactive_service is None:
        _proactive_service = ProactiveService()
    return _proactive_service
