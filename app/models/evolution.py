"""Evolution data models — StrategyUpdate, EvolutionCycle, and related types."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    MODEL_SELECTION = "model_selection"
    WORKFLOW_DESIGN = "workflow_design"
    CONTEXT_ASSEMBLY = "context_assembly"
    SKILL_INJECTION = "skill_injection"
    TOOL_SELECTION = "tool_selection"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Granular performance records ───────────────────────────────────────────────


class ModelPerformanceRecord(BaseModel):
    """How well a specific model performs on a specific task type."""

    model_name: str
    task_type: str
    executions: int
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_duration_ms: float
    avg_tokens_used: float = 0.0
    # Relative to other models for same task (populated by analysis)
    rank: int = 0
    is_best: bool = False


class WorkflowPerformanceRecord(BaseModel):
    """How well a specific workflow performs."""

    workflow_name: str
    task_type: str = "unknown"
    executions: int
    completion_rate: float = Field(ge=0.0, le=1.0)
    avg_steps: float = 0.0
    error_rate: float = Field(ge=0.0, le=1.0)
    # Composite score: weighted completion_rate - error_rate
    effectiveness_score: float = 0.0


class EvolutionRecommendation(BaseModel):
    """A human-readable, actionable recommendation from one analysis run."""

    title: str
    description: str
    strategy_updates: List["StrategyUpdate"] = Field(default_factory=list)
    expected_cumulative_improvement_percent: float = 0.0
    priority: int = Field(ge=1, le=5, default=3)  # 1 = highest
    auto_apply: bool = False


# ── Core strategy and cycle models ────────────────────────────────────────────


class StrategyUpdate(BaseModel):
    """A single proposed or applied change to a system strategy."""

    id: str
    strategy_type: StrategyType
    current_behavior: str
    proposed_behavior: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    expected_improvement_percent: float
    risk_level: RiskLevel
    validation_result: Optional[bool] = None
    applied: bool = False
    applied_at: Optional[datetime] = None
    rollback_instructions: Optional[str] = None
    # Snapshot of the previous state (used by rollback)
    previous_state: Optional[Dict[str, Any]] = None

    def is_auto_appliable(self, max_risk: str = "medium") -> bool:
        """Return True if this update's risk is within the allowed threshold."""
        order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
        allowed = RiskLevel(max_risk)
        return order[self.risk_level] <= order[allowed]


class EvolutionCycle(BaseModel):
    """Record of one complete evolution cycle."""

    id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    executions_analyzed: int = 0
    patterns_discovered: List[Dict[str, Any]] = Field(default_factory=list)
    strategies_generated: List[StrategyUpdate] = Field(default_factory=list)
    strategies_validated: List[StrategyUpdate] = Field(default_factory=list)
    strategies_applied: List[StrategyUpdate] = Field(default_factory=list)
    improvements_achieved: Dict[str, Any] = Field(default_factory=dict)
    rollback_performed: bool = False
    error: Optional[str] = None
    # Rich performance snapshots captured during this cycle
    model_performance: List[ModelPerformanceRecord] = Field(default_factory=list)
    workflow_performance: List[WorkflowPerformanceRecord] = Field(default_factory=list)
    recommendations: List[EvolutionRecommendation] = Field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.completed_at is not None and self.error is None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


# Resolve forward reference
EvolutionRecommendation.model_rebuild()
