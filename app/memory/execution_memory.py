"""Execution memory — tracks successful fixes, failed attempts, model performance."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ExecutionMemory:
    """Records what worked and what didn't across agent runs."""

    def __init__(self, data_dir: str = "data/memory") -> None:
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.executions_file = self.dir / "executions.json"
        self._executions: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.executions_file.exists():
            try:
                return json.loads(self.executions_file.read_text())
            except Exception:
                return []
        return []

    def _save(self) -> None:
        self.executions_file.write_text(json.dumps(self._executions, default=str, indent=2))

    def record(
        self,
        run_id: str,
        workflow: str,
        model: str,
        task_type: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._executions.append({
            "run_id": run_id,
            "workflow": workflow,
            "model": model,
            "task_type": task_type,
            "success": success,
            "duration_ms": duration_ms,
            "error": error,
            "metadata": metadata or {},
            "recorded_at": datetime.utcnow().isoformat(),
        })
        # Keep last 1000 executions
        self._executions = self._executions[-1000:]
        self._save()

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._executions[-limit:]

    def get_model_stats(self, model: str) -> Dict[str, Any]:
        runs = [e for e in self._executions if e["model"] == model]
        if not runs:
            return {"model": model, "runs": 0}
        successes = sum(1 for r in runs if r["success"])
        avg_ms = sum(r["duration_ms"] for r in runs) / len(runs)
        return {
            "model": model,
            "runs": len(runs),
            "success_rate": successes / len(runs),
            "avg_duration_ms": avg_ms,
        }

    def get_workflow_stats(self, workflow: str) -> Dict[str, Any]:
        runs = [e for e in self._executions if e["workflow"] == workflow]
        if not runs:
            return {"workflow": workflow, "runs": 0}
        successes = sum(1 for r in runs if r["success"])
        return {
            "workflow": workflow,
            "runs": len(runs),
            "success_rate": successes / len(runs),
        }


_memory: Optional[ExecutionMemory] = None


def get_execution_memory() -> ExecutionMemory:
    global _memory
    if _memory is None:
        _memory = ExecutionMemory()
    return _memory
