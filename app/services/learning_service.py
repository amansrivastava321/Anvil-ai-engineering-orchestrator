"""Learning service — records executions and builds performance recommendations."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

DATA_ROOT = Path("data/performance_db")


class LearningService:
    """Tracks execution performance and builds model/workflow recommendations."""

    def __init__(self, data_dir: str = "data/performance_db") -> None:
        self.root = Path(data_dir)
        for sub in ("executions", "patterns", "strategies", "feedback"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ── Record ─────────────────────────────────────────────────────────────

    def record_execution(self, run_data: Dict[str, Any]) -> None:
        run_id = run_data.get("run_id", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        path = self.root / "executions" / f"{run_id}.json"
        path.write_text(json.dumps({**run_data, "recorded_at": datetime.utcnow().isoformat()}, default=str, indent=2))
        logger.debug("Execution recorded", run_id=run_id)

    def score_execution(self, run_id: str, score: float) -> None:
        """Attach a human score (0.0–1.0) to an execution."""
        path = self.root / "feedback" / f"{run_id}.json"
        path.write_text(json.dumps({
            "run_id": run_id,
            "score": max(0.0, min(1.0, score)),
            "scored_at": datetime.utcnow().isoformat(),
        }, indent=2))

    # ── Recommend ──────────────────────────────────────────────────────────

    def recommend_model(self, task_type: str) -> Optional[str]:
        """Return the model with the best historical success rate for task_type."""
        stats: Dict[str, Dict[str, Any]] = {}
        for path in (self.root / "executions").glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("task_type") != task_type:
                    continue
                model = data.get("model_used")
                if not model:
                    continue
                if model not in stats:
                    stats[model] = {"total": 0, "success": 0}
                stats[model]["total"] += 1
                if data.get("status") == "completed":
                    stats[model]["success"] += 1
            except Exception:
                pass

        if not stats:
            return None

        best = max(stats.items(), key=lambda kv: kv[1]["success"] / max(kv[1]["total"], 1))
        return best[0]

    def recommend_workflow(self, task_type: str) -> Optional[str]:
        """Return the workflow with the best historical success rate for task_type."""
        stats: Dict[str, Dict[str, Any]] = {}
        for path in (self.root / "executions").glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("task_type") != task_type:
                    continue
                wf = data.get("workflow_type")
                if not wf:
                    continue
                if wf not in stats:
                    stats[wf] = {"total": 0, "success": 0}
                stats[wf]["total"] += 1
                if data.get("status") == "completed":
                    stats[wf]["success"] += 1
            except Exception:
                pass

        if not stats:
            return None

        best = max(stats.items(), key=lambda kv: kv[1]["success"] / max(kv[1]["total"], 1))
        return best[0]

    def get_stats(self) -> Dict[str, Any]:
        exec_count = len(list((self.root / "executions").glob("*.json")))
        fb_count = len(list((self.root / "feedback").glob("*.json")))
        return {"executions_recorded": exec_count, "feedback_given": fb_count}


_service: Optional[LearningService] = None


def get_learning_service() -> LearningService:
    global _service
    if _service is None:
        _service = LearningService()
    return _service
