"""Decision record — data models and persistence for organizational memory."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

_DECISIONS_DIR = Path("data/ai_decisions")
_DECISIONS_FILE = _DECISIONS_DIR / "decisions.json"


# ── Enums ────────────────────────────────────────────────────────────────────


class DecisionMode(str, Enum):
    DECIDE_ALONE = "decide_alone"
    CONSULT_EXPERTS = "consult_experts"
    CONVENE_COUNCIL = "convene_council"
    STRATEGIC_INITIATIVE = "strategic_initiative"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Core models ──────────────────────────────────────────────────────────────


class Problem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    systems_affected: List[str] = Field(default_factory=list)
    is_business_critical: bool = False
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def keywords(self) -> List[str]:
        """Extract lowercase words from description for pattern matching."""
        import re
        return re.findall(r"\b\w{3,}\b", self.description.lower())


class Proposal(BaseModel):
    """One council member's proposal for solving a problem."""
    member_name: str
    approach: str
    reasoning: str
    concerns: List[str] = Field(default_factory=list)
    recommended_tools: List[str] = Field(default_factory=list)
    estimated_risk: RiskLevel = RiskLevel.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class Critique(BaseModel):
    """One member's critique of another member's proposal."""
    critic_name: str
    target_member: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_considerations: List[str] = Field(default_factory=list)


class ElementVote(BaseModel):
    """Vote on a specific plan element (not whole proposals)."""
    element: str
    voter: str
    include: bool
    reasoning: str
    weight: float = Field(ge=0.0, le=1.0, default=1.0)


class SynthesizedPlan(BaseModel):
    """Unified plan produced by the Synthesizer from all proposals."""
    plan: str
    included_from: Dict[str, List[str]] = Field(default_factory=dict)
    rejected_elements: List[Dict[str, str]] = Field(default_factory=list)
    conflicts_resolved: List[Dict[str, str]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    needs_revision: bool = False


class CEODecision(BaseModel):
    """Complete record of one CEO decision, from problem receipt to learning."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    problem_id: str
    problem_description: str
    mode: DecisionMode
    mode_reasoning: str
    patterns_matched: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    experts_consulted: List[str] = Field(default_factory=list)
    final_plan: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    outcome_success: Optional[bool] = None
    outcome_notes: str = ""
    learning_notes: str = ""
    problem_keywords: List[str] = Field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.outcome_success is True


# ── Persistence ──────────────────────────────────────────────────────────────


class DecisionStore:
    """Persist and query CEO decision records."""

    def __init__(self, path: Path = _DECISIONS_FILE) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._decisions: List[CEODecision] = self._load()

    # ── public API ──────────────────────────────────────────────────────────

    def save(self, decision: CEODecision) -> None:
        self._decisions.append(decision)
        self._flush()

    def update(self, decision: CEODecision) -> None:
        for i, d in enumerate(self._decisions):
            if d.id == decision.id:
                self._decisions[i] = decision
                self._flush()
                return
        self.save(decision)

    def get(self, decision_id: str) -> Optional[CEODecision]:
        for d in self._decisions:
            if d.id == decision_id:
                return d
        return None

    def all(self, limit: int = 1000) -> List[CEODecision]:
        return list(reversed(self._decisions))[:limit]

    def successful(self, limit: int = 500) -> List[CEODecision]:
        return [d for d in self.all(limit * 2) if d.outcome_success is True][:limit]

    def failed(self, limit: int = 500) -> List[CEODecision]:
        return [d for d in self.all(limit * 2) if d.outcome_success is False][:limit]

    def count(self) -> int:
        return len(self._decisions)

    # ── internal ────────────────────────────────────────────────────────────

    def _load(self) -> List[CEODecision]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text())
            return [CEODecision.model_validate(d) for d in raw]
        except Exception:
            logger.warning("Failed to load decision records — starting fresh")
            return []

    def _flush(self) -> None:
        try:
            self._path.write_text(
                json.dumps([d.model_dump(mode="json") for d in self._decisions], indent=2)
            )
        except Exception as e:
            logger.error("Failed to persist decision records", error=str(e))


# ── Singleton ────────────────────────────────────────────────────────────────

_store: Optional[DecisionStore] = None


def get_decision_store() -> DecisionStore:
    global _store
    if _store is None:
        _store = DecisionStore()
    return _store
