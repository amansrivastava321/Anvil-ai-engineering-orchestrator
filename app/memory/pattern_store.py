"""Pattern store — records learned patterns from successful fixes and reviews."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class PatternStore:
    """Stores patterns learned from execution history."""

    def __init__(self, data_dir: str = "data/memory") -> None:
        self.path = Path(data_dir) / "patterns.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._patterns: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._patterns, default=str, indent=2))

    def record_pattern(
        self,
        pattern_type: str,
        pattern_key: str,
        context: Dict[str, Any],
        outcome: str,
    ) -> None:
        key = f"{pattern_type}:{pattern_key}"
        if key not in self._patterns:
            self._patterns[key] = {
                "type": pattern_type,
                "key": pattern_key,
                "outcomes": [],
                "first_seen": datetime.utcnow().isoformat(),
            }
        self._patterns[key]["outcomes"].append({
            "outcome": outcome,
            "context": context,
            "recorded_at": datetime.utcnow().isoformat(),
        })
        self._patterns[key]["last_seen"] = datetime.utcnow().isoformat()
        self._patterns[key]["count"] = len(self._patterns[key]["outcomes"])
        self._save()

    def get_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        patterns = list(self._patterns.values())
        if pattern_type:
            patterns = [p for p in patterns if p["type"] == pattern_type]
        return sorted(patterns, key=lambda p: p.get("count", 0), reverse=True)

    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Return the model with the best success rate for a task type."""
        # Placeholder: real impl would query execution_memory
        return None
