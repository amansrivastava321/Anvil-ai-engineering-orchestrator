"""Base memory store — JSON file-backed key/value store for agent memory."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class MemoryStore:
    """Lightweight JSON-backed memory store for agent execution history."""

    def __init__(self, store_path: str = "data/memory/memory.json") -> None:
        self.path = Path(store_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, default=str, indent=2))

    def set(self, key: str, value: Any) -> None:
        self._data[key] = {"value": value, "updated_at": datetime.utcnow().isoformat()}
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._data.get(key)
        return entry["value"] if entry else default

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._save()

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def all(self) -> Dict[str, Any]:
        return {k: v["value"] for k, v in self._data.items()}


_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
