"""Artifact store — persists every agent run to data/artifacts/runs/{run_id}/."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["ArtifactStore", "get_artifact_store"]


class ArtifactStore:
    """Stores and retrieves run artifacts from the local filesystem."""

    def __init__(self, base_path: str = "data/artifacts") -> None:
        self.base_path = Path(base_path)
        self.runs_path = self.base_path / "runs"
        self.runs_path.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self.runs_path / run_id

    def _new_run_id(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short = str(uuid.uuid4()).replace("-", "")[:8]
        return f"{ts}_{short}"

    # ── Write ─────────────────────────────────────────────────────────────────

    async def save_run(self, run_data: Dict[str, Any]) -> str:
        """Persist a run and return its run_id."""
        run_id = run_data.get("run_id") or self._new_run_id()
        run_dir = self._run_path(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # run.json — canonical record
        async with aiofiles.open(run_dir / "run.json", "w") as f:
            await f.write(json.dumps({**run_data, "run_id": run_id}, default=str, indent=2))

        # prompt.md
        if prompt := run_data.get("prompt"):
            async with aiofiles.open(run_dir / "prompt.md", "w") as f:
                await f.write(f"# Prompt\n\n{prompt}\n")

        # response.md
        if response := run_data.get("response"):
            async with aiofiles.open(run_dir / "response.md", "w") as f:
                await f.write(f"# Response\n\n{response}\n")

        # context.json
        if ctx := run_data.get("context"):
            async with aiofiles.open(run_dir / "context.json", "w") as f:
                await f.write(json.dumps(ctx, default=str, indent=2))

        # logs.txt
        if logs := run_data.get("logs"):
            async with aiofiles.open(run_dir / "logs.txt", "w") as f:
                await f.write(logs if isinstance(logs, str) else "\n".join(logs))

        # test_results.json
        if tests := run_data.get("test_results"):
            async with aiofiles.open(run_dir / "test_results.json", "w") as f:
                await f.write(json.dumps(tests, default=str, indent=2))

        # patches.diff
        if diff := run_data.get("patches"):
            async with aiofiles.open(run_dir / "patches.diff", "w") as f:
                await f.write(diff)

        logger.info("Run saved", run_id=run_id, path=str(run_dir))
        return run_id

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a run by ID. Returns None if not found."""
        run_file = self._run_path(run_id) / "run.json"
        if not run_file.exists():
            return None
        async with aiofiles.open(run_file) as f:
            return json.loads(await f.read())

    async def list_runs(
        self,
        repo_path: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List runs, optionally filtered by repo_path, newest first."""
        runs: List[Dict[str, Any]] = []
        if not self.runs_path.exists():
            return runs

        dirs = sorted(
            [d for d in self.runs_path.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )

        for run_dir in dirs:
            if len(runs) >= limit:
                break
            run_file = run_dir / "run.json"
            if not run_file.exists():
                continue
            try:
                async with aiofiles.open(run_file) as f:
                    data = json.loads(await f.read())
                if repo_path and data.get("repo_path") != repo_path:
                    continue
                # Return summary (not full context)
                runs.append({
                    "run_id": data.get("run_id", run_dir.name),
                    "repo_path": data.get("repo_path"),
                    "prompt": (data.get("prompt") or "")[:120],
                    "workflow": data.get("workflow_type"),
                    "status": data.get("status"),
                    "timestamp": data.get("timestamp"),
                    "duration_ms": data.get("duration_ms"),
                })
            except Exception:
                continue

        return runs

    async def search_runs(self, query: str) -> List[Dict[str, Any]]:
        """Search runs by prompt or response content."""
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []

        if not self.runs_path.exists():
            return results

        for run_dir in sorted(self.runs_path.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            run_file = run_dir / "run.json"
            if not run_file.exists():
                continue
            try:
                async with aiofiles.open(run_file) as f:
                    data = json.loads(await f.read())
                prompt = (data.get("prompt") or "").lower()
                response = (data.get("response") or "").lower()
                if query_lower in prompt or query_lower in response:
                    results.append({
                        "run_id": data.get("run_id", run_dir.name),
                        "repo_path": data.get("repo_path"),
                        "prompt": (data.get("prompt") or "")[:120],
                        "workflow": data.get("workflow_type"),
                        "timestamp": data.get("timestamp"),
                    })
            except Exception:
                continue

        return results

    # ── Legacy compat (orchestrator may call these) ───────────────────────────

    async def save_artifact(
        self,
        execution_id: str,
        artifact_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Legacy: save a single artifact. Delegates to save_run."""
        return await self.save_run({
            "run_id": execution_id,
            "artifact_type": artifact_type,
            "response": str(content),
            **(metadata or {}),
        })

    async def get_artifact(self, execution_id: str, artifact_type: str) -> Optional[Dict[str, Any]]:
        """Legacy: get artifact. Delegates to get_run."""
        return await self.get_run(execution_id)

    async def list_artifacts(self, execution_id: str) -> List[Dict[str, Any]]:
        """Legacy: list artifacts for an execution."""
        run = await self.get_run(execution_id)
        return [run] if run else []

    async def close(self) -> None:
        pass


_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Return the process-wide ArtifactStore singleton."""
    global _store
    if _store is None:
        _store = ArtifactStore()
    return _store
