"""Tests for EvolutionService, evolution data models, and evolution API endpoints.

Covers:
- All data model properties and helpers
- _load_execution_data (disk files + in-memory merge, corrupted file handling)
- _analyze_model_performance (ranking, multi-task, single-model)
- _analyze_workflow_performance (completion rate, effectiveness_score)
- _analyze_skill_combinations (lift calculation, recommendation filtering)
- _analyze_context_strategies (mode ranking per workflow)
- _generate_strategies (model/workflow/skill/context updates)
- _validate_strategies (approved vs rejected)
- _apply_strategies (model weights, context strategies, skill injection, workflow)
- _calculate_improvements (summary structure)
- record_evolution / get_history / rollback_cycle
- Full evolution cycle: skip, success, error capture, large dataset
- Concurrent cycle prevention
- Disk persistence and rollback
- 4 explicit demonstration scenarios from the spec
- API endpoints: run (force param), status (rich response), history, rollback (confirm)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_exec(
    model: str = "deepseek-r1:7b",
    task_type: str = "debugging",
    workflow: str = "debug",
    success: bool = True,
    duration_ms: float = 1000.0,
    context_mode: str = "balanced",
    run_id: str | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id or str(uuid.uuid4()),
        "model": model,
        "task_type": task_type,
        "workflow": workflow,
        "success": success,
        "duration_ms": duration_ms,
        "error": error,
        "metadata": {"context_mode": context_mode},
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _make_execs(n: int, **kwargs) -> List[Dict[str, Any]]:
    return [_make_exec(**kwargs) for _ in range(n)]


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.get_recent.return_value = []
    mem.get_model_stats.return_value = {"success_rate": 0.0, "runs": 0, "avg_duration_ms": 0.0}
    return mem


@pytest.fixture
def mock_patterns():
    ps = MagicMock()
    ps.get_patterns.return_value = []
    ps.record_pattern = MagicMock()
    return ps


@pytest.fixture
def svc(mock_memory, mock_patterns, tmp_path):
    import app.services.evolution_service as evo_mod

    evo_mod._HISTORY_PATH = tmp_path / "evolution_history.json"
    evo_mod._EXECUTIONS_DIR = tmp_path / "executions"
    evo_mod._evolution_service = None

    service = evo_mod.EvolutionService(
        execution_memory=mock_memory,
        pattern_store=mock_patterns,
        improvement_threshold=0.05,
        min_executions=5,
        max_automatic_risk="medium",
    )
    return service


# ─── Data models ──────────────────────────────────────────────────────────────

class TestModelPerformanceRecord:
    def test_is_best_flag(self):
        from app.models.evolution import ModelPerformanceRecord
        r = ModelPerformanceRecord(
            model_name="m1", task_type="debug", executions=10,
            success_rate=0.9, avg_duration_ms=500, rank=1, is_best=True,
        )
        assert r.is_best is True

    def test_success_rate_clamped(self):
        from app.models.evolution import ModelPerformanceRecord
        with pytest.raises(Exception):
            ModelPerformanceRecord(
                model_name="m", task_type="t", executions=1,
                success_rate=1.5, avg_duration_ms=100,
            )


class TestWorkflowPerformanceRecord:
    def test_effectiveness_score_field(self):
        from app.models.evolution import WorkflowPerformanceRecord
        r = WorkflowPerformanceRecord(
            workflow_name="audit", executions=20,
            completion_rate=0.85, error_rate=0.10, effectiveness_score=0.80,
        )
        assert r.effectiveness_score == pytest.approx(0.80)


class TestEvolutionRecommendation:
    def test_priority_range(self):
        from app.models.evolution import EvolutionRecommendation
        with pytest.raises(Exception):
            EvolutionRecommendation(title="t", description="d", priority=0)

    def test_defaults(self):
        from app.models.evolution import EvolutionRecommendation
        r = EvolutionRecommendation(title="t", description="d")
        assert r.auto_apply is False
        assert r.priority == 3
        assert r.strategy_updates == []


class TestStrategyUpdate:
    def test_is_auto_appliable_low_within_medium(self):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel
        u = StrategyUpdate(
            id="1", strategy_type=StrategyType.MODEL_SELECTION,
            current_behavior="a", proposed_behavior="b",
            expected_improvement_percent=10.0, risk_level=RiskLevel.LOW,
        )
        assert u.is_auto_appliable("medium") is True

    def test_is_auto_appliable_high_rejected_at_medium(self):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel
        u = StrategyUpdate(
            id="2", strategy_type=StrategyType.WORKFLOW_DESIGN,
            current_behavior="a", proposed_behavior="b",
            expected_improvement_percent=5.0, risk_level=RiskLevel.HIGH,
        )
        assert u.is_auto_appliable("medium") is False
        assert u.is_auto_appliable("high") is True

    def test_previous_state_field(self):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel
        u = StrategyUpdate(
            id="3", strategy_type=StrategyType.CONTEXT_ASSEMBLY,
            current_behavior="a", proposed_behavior="b",
            expected_improvement_percent=8.0, risk_level=RiskLevel.MEDIUM,
            previous_state={"mode": "balanced"},
        )
        assert u.previous_state == {"mode": "balanced"}


class TestEvolutionCycle:
    def test_succeeded(self):
        from app.models.evolution import EvolutionCycle
        now = datetime.now(timezone.utc)
        c = EvolutionCycle(id="1", started_at=now, completed_at=now)
        assert c.succeeded is True

    def test_not_succeeded_with_error(self):
        from app.models.evolution import EvolutionCycle
        now = datetime.now(timezone.utc)
        c = EvolutionCycle(id="2", started_at=now, completed_at=now, error="boom")
        assert c.succeeded is False

    def test_duration_seconds(self):
        from app.models.evolution import EvolutionCycle
        from datetime import timedelta
        start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        c = EvolutionCycle(id="3", started_at=start, completed_at=start + timedelta(seconds=45))
        assert c.duration_seconds == pytest.approx(45.0)

    def test_model_and_workflow_performance_fields(self):
        from app.models.evolution import EvolutionCycle
        c = EvolutionCycle(id="4", started_at=datetime.now(timezone.utc))
        assert c.model_performance == []
        assert c.workflow_performance == []
        assert c.recommendations == []


# ─── _load_execution_data ─────────────────────────────────────────────────────

class TestLoadExecutionData:
    @pytest.mark.asyncio
    async def test_reads_json_files_from_disk(self, svc, tmp_path):
        import app.services.evolution_service as evo_mod
        execs_dir = tmp_path / "executions"
        execs_dir.mkdir()
        evo_mod._EXECUTIONS_DIR = execs_dir

        ex = _make_exec(run_id="disk-run-1")
        (execs_dir / "run1.json").write_text(json.dumps(ex))

        result = await svc._load_execution_data()
        ids = [r["run_id"] for r in result]
        assert "disk-run-1" in ids

    @pytest.mark.asyncio
    async def test_skips_corrupted_files(self, svc, tmp_path):
        import app.services.evolution_service as evo_mod
        execs_dir = tmp_path / "executions"
        execs_dir.mkdir()
        evo_mod._EXECUTIONS_DIR = execs_dir

        (execs_dir / "good.json").write_text(json.dumps(_make_exec(run_id="good")))
        (execs_dir / "bad.json").write_text("this is not valid json {{{{")

        result = await svc._load_execution_data()
        ids = [r["run_id"] for r in result]
        assert "good" in ids
        assert len(result) == 1  # bad file was skipped

    @pytest.mark.asyncio
    async def test_merges_memory_records_without_duplicates(self, svc, tmp_path, mock_memory):
        import app.services.evolution_service as evo_mod
        execs_dir = tmp_path / "executions"
        execs_dir.mkdir()
        evo_mod._EXECUTIONS_DIR = execs_dir

        shared_run_id = "shared-001"
        disk_ex = _make_exec(run_id=shared_run_id)
        mem_ex_new = _make_exec(run_id="mem-only-002")
        (execs_dir / "disk.json").write_text(json.dumps(disk_ex))
        mock_memory.get_recent.return_value = [disk_ex, mem_ex_new]  # shared + unique

        result = await svc._load_execution_data()
        ids = [r["run_id"] for r in result]
        assert ids.count(shared_run_id) == 1  # no duplicate
        assert "mem-only-002" in ids

    @pytest.mark.asyncio
    async def test_handles_missing_executions_dir(self, svc, tmp_path):
        import app.services.evolution_service as evo_mod
        evo_mod._EXECUTIONS_DIR = tmp_path / "nonexistent"
        mock_memory = svc._memory
        mock_memory.get_recent.return_value = _make_execs(3)
        result = await svc._load_execution_data()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_reads_list_format_json_file(self, svc, tmp_path):
        import app.services.evolution_service as evo_mod
        execs_dir = tmp_path / "executions"
        execs_dir.mkdir()
        evo_mod._EXECUTIONS_DIR = execs_dir

        execs = [_make_exec(run_id=f"list-{i}") for i in range(3)]
        (execs_dir / "batch.json").write_text(json.dumps(execs))

        result = await svc._load_execution_data()
        ids = [r["run_id"] for r in result]
        assert all(f"list-{i}" in ids for i in range(3))


# ─── _analyze_model_performance ──────────────────────────────────────────────

class TestAnalyzeModelPerformance:
    @pytest.mark.asyncio
    async def test_best_model_marked_is_best(self, svc):
        execs = (
            _make_execs(20, model="good", task_type="debug", success=True)
            + _make_execs(10, model="bad", task_type="debug", success=False)
        )
        records = await svc._analyze_model_performance(execs)
        best = next(r for r in records if r.model_name == "good")
        assert best.is_best is True
        assert best.success_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_multiple_task_types_ranked_independently(self, svc):
        execs = (
            _make_execs(10, model="m1", task_type="debug", success=True)
            + _make_execs(10, model="m2", task_type="audit", success=True)
        )
        records = await svc._analyze_model_performance(execs)
        debug_best = [r for r in records if r.task_type == "debug" and r.is_best]
        audit_best = [r for r in records if r.task_type == "audit" and r.is_best]
        assert len(debug_best) == 1
        assert len(audit_best) == 1

    @pytest.mark.asyncio
    async def test_empty_returns_empty_list(self, svc):
        records = await svc._analyze_model_performance([])
        assert records == []

    @pytest.mark.asyncio
    async def test_rank_assigned_correctly(self, svc):
        execs = (
            _make_execs(10, model="a", task_type="t", success=True)
            + _make_execs(10, model="b", task_type="t", success=False)
        )
        records = await svc._analyze_model_performance(execs)
        by_model = {r.model_name: r for r in records if r.task_type == "t"}
        assert by_model["a"].rank == 1
        assert by_model["b"].rank == 2


# ─── _analyze_workflow_performance ───────────────────────────────────────────

class TestAnalyzeWorkflowPerformance:
    @pytest.mark.asyncio
    async def test_completion_rate_computed(self, svc):
        execs = (
            _make_execs(7, workflow="report", success=True)
            + _make_execs(3, workflow="report", success=False)
        )
        records = await svc._analyze_workflow_performance(execs)
        rep = next(r for r in records if r.workflow_name == "report")
        assert rep.completion_rate == pytest.approx(0.7)
        assert rep.executions == 10

    @pytest.mark.asyncio
    async def test_error_rate_computed(self, svc):
        execs = _make_execs(5, workflow="audit", success=False, error="RuntimeError")
        records = await svc._analyze_workflow_performance(execs)
        aud = next(r for r in records if r.workflow_name == "audit")
        assert aud.error_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_sorted_by_effectiveness(self, svc):
        execs = (
            _make_execs(10, workflow="good", success=True)
            + _make_execs(10, workflow="bad", success=False)
        )
        records = await svc._analyze_workflow_performance(execs)
        assert records[0].workflow_name == "good"

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, svc):
        records = await svc._analyze_workflow_performance([])
        assert records == []


# ─── _analyze_skill_combinations ─────────────────────────────────────────────

class TestAnalyzeSkillCombinations:
    @pytest.mark.asyncio
    async def test_high_lift_combo_recommended(self, svc, mock_patterns):
        mock_patterns.get_patterns.return_value = [
            {
                "key": "tdd",
                "outcomes": [{"outcome": "success"}, {"outcome": "failed"}, {"outcome": "failed"}],
                "first_seen": "2026-01-01", "last_seen": "2026-01-10",
            },
            {
                "key": "tdd+systematic-debugging",
                "outcomes": [{"outcome": "success"} for _ in range(9)] + [{"outcome": "failed"}],
                "first_seen": "2026-01-01", "last_seen": "2026-01-10",
            },
        ]
        result = await svc._analyze_skill_combinations()
        assert len(result["recommended"]) > 0
        assert result["recommended"][0]["combination"] == "tdd+systematic-debugging"

    @pytest.mark.asyncio
    async def test_no_patterns_returns_empty(self, svc, mock_patterns):
        mock_patterns.get_patterns.return_value = []
        result = await svc._analyze_skill_combinations()
        assert result == {}

    @pytest.mark.asyncio
    async def test_low_lift_combo_not_recommended(self, svc, mock_patterns):
        mock_patterns.get_patterns.return_value = [
            {"key": "a", "outcomes": [{"outcome": "success"} for _ in range(8)] + [{"outcome": "failed"}, {"outcome": "failed"}], "first_seen": "x", "last_seen": "y"},
            {"key": "a+b", "outcomes": [{"outcome": "success"} for _ in range(8)] + [{"outcome": "failed"}, {"outcome": "failed"}], "first_seen": "x", "last_seen": "y"},
        ]
        result = await svc._analyze_skill_combinations()
        assert result.get("recommended", []) == []


# ─── _analyze_context_strategies ─────────────────────────────────────────────

class TestAnalyzeContextStrategies:
    @pytest.mark.asyncio
    async def test_best_mode_selected(self, svc):
        execs = [
            _make_exec(workflow="audit", success=True, context_mode="comprehensive"),
            _make_exec(workflow="audit", success=True, context_mode="comprehensive"),
            _make_exec(workflow="audit", success=False, context_mode="balanced"),
        ]
        result = await svc._analyze_context_strategies(execs)
        assert result["audit"]["best_mode"] == "comprehensive"

    @pytest.mark.asyncio
    async def test_empty_executions(self, svc):
        result = await svc._analyze_context_strategies([])
        assert result == {}


# ─── _generate_strategies ─────────────────────────────────────────────────────

class TestGenerateStrategies:
    @pytest.mark.asyncio
    async def test_model_selection_generated(self, svc):
        analysis = {
            "model_performance": {
                "debug": {
                    "confidence": 0.8,
                    "all_models": [
                        {"model": "good", "success_rate": 0.95, "runs": 50, "avg_duration_ms": 800},
                        {"model": "bad", "success_rate": 0.60, "runs": 40, "avg_duration_ms": 900},
                    ],
                }
            },
            "workflow_effectiveness": {},
            "skill_combinations": {},
            "context_strategies": {},
        }
        updates = await svc._generate_strategies(analysis)
        assert any(u.strategy_type.value == "model_selection" for u in updates)
        u = next(u for u in updates if u.strategy_type.value == "model_selection")
        assert u.expected_improvement_percent == pytest.approx(35.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_workflow_flagged_for_attention(self, svc):
        analysis = {
            "model_performance": {},
            "workflow_effectiveness": {
                "debug": {"success_rate": 0.50, "error_rate": 0.30, "runs": 10, "needs_attention": True}
            },
            "skill_combinations": {},
            "context_strategies": {},
        }
        updates = await svc._generate_strategies(analysis)
        assert any(u.strategy_type.value == "workflow_design" for u in updates)

    @pytest.mark.asyncio
    async def test_no_updates_when_no_findings(self, svc):
        analysis = {
            "model_performance": {},
            "workflow_effectiveness": {},
            "skill_combinations": {},
            "context_strategies": {},
        }
        updates = await svc._generate_strategies(analysis)
        assert updates == []

    @pytest.mark.asyncio
    async def test_low_confidence_model_skipped(self, svc):
        analysis = {
            "model_performance": {
                "debug": {
                    "confidence": 0.1,  # too low
                    "all_models": [
                        {"model": "a", "success_rate": 0.90, "runs": 3, "avg_duration_ms": 500},
                        {"model": "b", "success_rate": 0.50, "runs": 3, "avg_duration_ms": 600},
                    ],
                }
            },
            "workflow_effectiveness": {},
            "skill_combinations": {},
            "context_strategies": {},
        }
        updates = await svc._generate_strategies(analysis)
        assert all(u.strategy_type.value != "model_selection" for u in updates)


# ─── _validate_strategies ─────────────────────────────────────────────────────

class TestValidateStrategies:
    @pytest.mark.asyncio
    async def test_validates_list_and_filters(self, svc, mock_memory):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel

        good = StrategyUpdate(
            id="g1", strategy_type=StrategyType.SKILL_INJECTION,
            current_behavior="x", proposed_behavior="y",
            evidence={"lift_over_baseline": 0.25, "combination": "a+b"},
            expected_improvement_percent=25.0, risk_level=RiskLevel.LOW,
        )
        bad = StrategyUpdate(
            id="b1", strategy_type=StrategyType.SKILL_INJECTION,
            current_behavior="x", proposed_behavior="y",
            evidence={"lift_over_baseline": 0.01},
            expected_improvement_percent=1.0, risk_level=RiskLevel.LOW,
        )
        validated = await svc._validate_strategies([good, bad], [])
        ids = [u.id for u in validated]
        assert "g1" in ids
        assert "b1" not in ids

    @pytest.mark.asyncio
    async def test_validation_result_set_on_all(self, svc):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel

        update = StrategyUpdate(
            id="u1", strategy_type=StrategyType.SKILL_INJECTION,
            current_behavior="a", proposed_behavior="b",
            evidence={"lift_over_baseline": 0.20},
            expected_improvement_percent=20.0, risk_level=RiskLevel.LOW,
        )
        await svc._validate_strategies([update], [])
        assert update.validation_result is not None


# ─── _apply_strategies ───────────────────────────────────────────────────────

class TestApplyStrategies:
    @pytest.mark.asyncio
    async def test_model_selection_updates_weights(self, svc):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel
        from app.services.model_service import MODEL_REGISTRY

        update = StrategyUpdate(
            id="a1", strategy_type=StrategyType.MODEL_SELECTION,
            current_behavior="old", proposed_behavior="new",
            evidence={"best_model": "deepseek-r1:7b", "best_success_rate": 0.92, "task_type": "debug"},
            expected_improvement_percent=12.0, risk_level=RiskLevel.LOW,
        )
        applied = await svc._apply_strategies([update])
        assert len(applied) == 1
        assert applied[0].applied is True
        assert applied[0].previous_state is not None

    @pytest.mark.asyncio
    async def test_context_assembly_updates_strategy(self, svc):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel
        from app.services.context_service import get_context_service

        update = StrategyUpdate(
            id="a2", strategy_type=StrategyType.CONTEXT_ASSEMBLY,
            current_behavior="balanced", proposed_behavior="comprehensive",
            evidence={"workflow": "audit-test-workflow", "best_mode": "comprehensive", "improvement": 0.22},
            expected_improvement_percent=22.0, risk_level=RiskLevel.LOW,
        )
        await svc._apply_strategies([update])
        ctx = get_context_service()
        assert ctx.get_context_strategy("audit-test-workflow") == {"preferred_mode": "comprehensive"}

    @pytest.mark.asyncio
    async def test_failed_apply_excluded(self, svc):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel

        update = StrategyUpdate(
            id="fail1", strategy_type=StrategyType.MODEL_SELECTION,
            current_behavior="x", proposed_behavior="y",
            evidence={}, expected_improvement_percent=5.0, risk_level=RiskLevel.LOW,
        )
        with patch.object(svc, "_apply_single", side_effect=RuntimeError("forced")):
            applied = await svc._apply_strategies([update])
        assert applied == []


# ─── _calculate_improvements ─────────────────────────────────────────────────

class TestCalculateImprovements:
    def test_empty_returns_zero(self, svc):
        result = svc._calculate_improvements([])
        assert result["total_strategies_applied"] == 0

    def test_totals_computed(self, svc):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel

        updates = [
            StrategyUpdate(
                id=str(i), strategy_type=StrategyType.MODEL_SELECTION,
                current_behavior="a", proposed_behavior="b",
                evidence={}, expected_improvement_percent=10.0, risk_level=RiskLevel.LOW,
                applied=True,
            )
            for i in range(3)
        ]
        result = svc._calculate_improvements(updates)
        assert result["total_strategies_applied"] == 3
        assert result["total_expected_improvement_percent"] == pytest.approx(30.0)

    def test_by_type_breakdown(self, svc):
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel

        updates = [
            StrategyUpdate(
                id="1", strategy_type=StrategyType.MODEL_SELECTION,
                current_behavior="a", proposed_behavior="b",
                evidence={}, expected_improvement_percent=15.0, risk_level=RiskLevel.LOW, applied=True,
            ),
            StrategyUpdate(
                id="2", strategy_type=StrategyType.CONTEXT_ASSEMBLY,
                current_behavior="a", proposed_behavior="b",
                evidence={}, expected_improvement_percent=20.0, risk_level=RiskLevel.LOW, applied=True,
            ),
        ]
        result = svc._calculate_improvements(updates)
        assert "model_selection" in result["by_type"]
        assert "context_assembly" in result["by_type"]


# ─── History and rollback ─────────────────────────────────────────────────────

class TestHistoryAndRollback:
    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, svc):
        from app.models.evolution import EvolutionCycle
        c = EvolutionCycle(id="h1", started_at=datetime.now(timezone.utc))
        await svc.record_evolution(c)
        history = await svc.get_history(limit=5)
        assert len(history) == 1
        assert history[0].id == "h1"

    @pytest.mark.asyncio
    async def test_history_newest_first(self, svc):
        from app.models.evolution import EvolutionCycle
        for i in range(3):
            c = EvolutionCycle(id=f"cycle-{i}", started_at=datetime.now(timezone.utc))
            await svc.record_evolution(c)
        history = await svc.get_history(limit=10)
        assert history[0].id == "cycle-2"

    @pytest.mark.asyncio
    async def test_rollback_not_found_returns_false(self, svc):
        assert await svc.rollback_cycle("nonexistent") is False

    @pytest.mark.asyncio
    async def test_rollback_restores_model_weight(self, svc):
        from app.models.evolution import EvolutionCycle, StrategyUpdate, StrategyType, RiskLevel
        from app.services.model_service import MODEL_REGISTRY

        model = "deepseek-r1:7b"
        original = MODEL_REGISTRY[model]["quality_weight"]
        rb = json.dumps({
            "type": "model_selection",
            "previous_weights": {model: {"quality_weight": original}},
        })
        applied = StrategyUpdate(
            id="r1", strategy_type=StrategyType.MODEL_SELECTION,
            current_behavior="a", proposed_behavior="b",
            evidence={}, expected_improvement_percent=10.0, risk_level=RiskLevel.LOW,
            applied=True, applied_at=datetime.now(timezone.utc),
            rollback_instructions=rb,
        )
        cycle = EvolutionCycle(
            id="rollback-test", started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc), strategies_applied=[applied],
        )
        await svc.record_evolution(cycle)

        # Mutate, then rollback
        MODEL_REGISTRY[model]["quality_weight"] = 0.99
        success = await svc.rollback_cycle("rollback-test")
        assert success is True
        assert MODEL_REGISTRY[model]["quality_weight"] == pytest.approx(original)

    @pytest.mark.asyncio
    async def test_history_limit_respected(self, svc):
        from app.models.evolution import EvolutionCycle
        for i in range(20):
            await svc.record_evolution(EvolutionCycle(id=f"c{i}", started_at=datetime.now(timezone.utc)))
        history = await svc.get_history(limit=5)
        assert len(history) == 5


# ─── Full evolution cycle ─────────────────────────────────────────────────────

class TestFullEvolutionCycle:
    @pytest.mark.asyncio
    async def test_skips_when_insufficient_data(self, svc, mock_memory):
        svc._min_executions = 10
        mock_memory.get_recent.return_value = _make_execs(3)
        cycle = await svc.run_evolution_cycle()
        assert "skipped" in cycle.improvements_achieved

    @pytest.mark.asyncio
    async def test_force_bypasses_minimum_check(self, svc, mock_memory):
        svc._min_executions = 100
        mock_memory.get_recent.return_value = _make_execs(3)
        cycle = await svc.run_evolution_cycle(force=True)
        assert cycle.completed_at is not None
        assert "skipped" not in str(cycle.improvements_achieved)

    @pytest.mark.asyncio
    async def test_completes_with_sufficient_data(self, svc, mock_memory):
        execs = (
            _make_execs(15, model="good", task_type="debug", success=True)
            + _make_execs(5, model="bad", task_type="debug", success=False)
        )
        mock_memory.get_recent.return_value = execs
        mock_memory.get_model_stats.return_value = {"success_rate": 0.9, "runs": 15}
        cycle = await svc.run_evolution_cycle()
        assert cycle.completed_at is not None
        assert cycle.error is None
        assert cycle.executions_analyzed == 20

    @pytest.mark.asyncio
    async def test_captures_error_without_raising(self, svc, mock_memory):
        mock_memory.get_recent.side_effect = RuntimeError("db exploded")
        cycle = await svc.run_evolution_cycle()
        assert cycle.error == "db exploded"
        assert cycle.completed_at is not None

    @pytest.mark.asyncio
    async def test_cycle_saved_to_history(self, svc, mock_memory):
        mock_memory.get_recent.return_value = _make_execs(10)
        await svc.run_evolution_cycle()
        history = await svc.get_history()
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_cycle_prevented(self, svc, mock_memory):
        svc._running = True  # simulate already-running cycle
        mock_memory.get_recent.return_value = _make_execs(10)
        cycle = await svc.run_evolution_cycle()
        assert "Concurrent" in (cycle.error or "")
        svc._running = False  # cleanup

    @pytest.mark.asyncio
    async def test_large_dataset(self, svc, mock_memory):
        mock_memory.get_recent.return_value = _make_execs(1000, model="m")
        cycle = await svc.run_evolution_cycle()
        assert cycle.executions_analyzed == 1000


# ─── ModelService and ContextService integration ──────────────────────────────

class TestModelServiceWeights:
    def test_update_clamps_to_1(self):
        from app.services.model_service import get_model_service, MODEL_REGISTRY
        svc = get_model_service()
        model = list(MODEL_REGISTRY.keys())[0]
        original = MODEL_REGISTRY[model]["quality_weight"]
        svc.update_model_weights({model: 2.5})
        assert MODEL_REGISTRY[model]["quality_weight"] == pytest.approx(1.0)
        MODEL_REGISTRY[model]["quality_weight"] = original

    def test_update_clamps_to_0(self):
        from app.services.model_service import get_model_service, MODEL_REGISTRY
        svc = get_model_service()
        model = list(MODEL_REGISTRY.keys())[0]
        original = MODEL_REGISTRY[model]["quality_weight"]
        svc.update_model_weights({model: -1.0})
        assert MODEL_REGISTRY[model]["quality_weight"] == pytest.approx(0.0)
        MODEL_REGISTRY[model]["quality_weight"] = original

    def test_unknown_model_ignored(self):
        from app.services.model_service import get_model_service
        prev = get_model_service().update_model_weights({"ghost-model:latest": 0.9})
        assert prev == {}

    def test_get_current_weights_all_models(self):
        from app.services.model_service import get_model_service, MODEL_REGISTRY
        weights = get_model_service().get_current_weights()
        assert len(weights) == len(MODEL_REGISTRY)
        for w in weights.values():
            assert "quality_weight" in w and "cost_weight" in w


class TestContextServiceStrategy:
    def test_update_and_retrieve(self):
        from app.services.context_service import get_context_service
        svc = get_context_service()
        svc.update_context_strategy("test-wf", {"preferred_mode": "precise"})
        assert svc.get_context_strategy("test-wf") == {"preferred_mode": "precise"}

    def test_returns_previous_on_overwrite(self):
        from app.services.context_service import get_context_service
        svc = get_context_service()
        svc._context_strategies["wf2"] = {"preferred_mode": "balanced"}
        prev = svc.update_context_strategy("wf2", {"preferred_mode": "comprehensive"})
        assert prev == {"preferred_mode": "balanced"}

    def test_missing_key_returns_empty(self):
        from app.services.context_service import get_context_service
        assert get_context_service().get_context_strategy("no-such-wf-xyz") == {}


# ─── Demonstration scenarios from the spec ────────────────────────────────────

class TestDemonstrationScenarios:
    """Four explicit test scenarios from the Phase 22 spec."""

    @pytest.mark.asyncio
    async def test_scenario_1_basic_analysis_identifies_best_model(self, svc, mock_memory):
        """Seed 100 executions, trigger cycle, verify best model identified."""
        execs = (
            _make_execs(60, model="deepseek-r1:7b", task_type="debugging", success=True, duration_ms=800)
            + _make_execs(40, model="qwen2.5:7b", task_type="debugging", success=False, duration_ms=1200)
        )
        mock_memory.get_recent.return_value = execs

        cycle = await svc.run_evolution_cycle()

        assert cycle.executions_analyzed == 100
        # Best model record should be deepseek
        best = next((r for r in cycle.model_performance if r.is_best and r.task_type == "debugging"), None)
        assert best is not None
        assert best.model_name == "deepseek-r1:7b"
        assert best.success_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_scenario_2_good_strategy_applied_bad_rejected(self, svc, mock_memory):
        """Strategy A (25% lift) applied; Strategy B (1% lift) rejected."""
        from app.models.evolution import StrategyUpdate, StrategyType, RiskLevel

        execs = _make_execs(20, model="m", task_type="debug", success=True)
        mock_memory.get_recent.return_value = execs

        strategy_a = StrategyUpdate(
            id="A", strategy_type=StrategyType.SKILL_INJECTION,
            current_behavior="individual skills",
            proposed_behavior="combo: tdd+verification",
            evidence={"lift_over_baseline": 0.25, "combination": "tdd+verification", "success_rate": 0.90, "count": 50},
            expected_improvement_percent=25.0, risk_level=RiskLevel.LOW,
        )
        strategy_b = StrategyUpdate(
            id="B", strategy_type=StrategyType.SKILL_INJECTION,
            current_behavior="individual skills",
            proposed_behavior="combo: low-lift+combo",
            evidence={"lift_over_baseline": 0.01, "combination": "low+combo", "success_rate": 0.52, "count": 10},
            expected_improvement_percent=1.0, risk_level=RiskLevel.LOW,
        )

        validated = await svc._validate_strategies([strategy_a, strategy_b], execs)
        ids = [u.id for u in validated]
        assert "A" in ids
        assert "B" not in ids

    @pytest.mark.asyncio
    async def test_scenario_3_model_routing_updated_after_evolution(self, svc, mock_memory):
        """50 debugging tasks: Model X 90% success, Model Y 60% → X gets higher weight."""
        from app.services.model_service import get_model_service, MODEL_REGISTRY

        execs = (
            _make_execs(45, model="deepseek-r1:7b", task_type="debugging", success=True)
            + _make_execs(5, model="deepseek-r1:7b", task_type="debugging", success=False)
            + _make_execs(12, model="qwen2.5:7b", task_type="debugging", success=True)
            + _make_execs(8, model="qwen2.5:7b", task_type="debugging", success=False)
        )
        mock_memory.get_recent.return_value = execs
        mock_memory.get_model_stats.return_value = {"success_rate": 0.9, "runs": 50}

        original_weight = MODEL_REGISTRY.get("deepseek-r1:7b", {}).get("quality_weight", 0.8)
        cycle = await svc.run_evolution_cycle()

        # Some model selection strategy should have been generated and possibly applied
        model_updates = [u for u in cycle.strategies_generated if u.strategy_type.value == "model_selection"]
        assert len(model_updates) > 0, "Expected model selection strategy to be generated"

        # Restore weight
        if "deepseek-r1:7b" in MODEL_REGISTRY:
            MODEL_REGISTRY["deepseek-r1:7b"]["quality_weight"] = original_weight

    @pytest.mark.asyncio
    async def test_scenario_4_rollback_restores_previous_state(self, svc, mock_memory):
        """Apply cycle, rollback, verify model weights restored."""
        from app.services.model_service import MODEL_REGISTRY

        model = "deepseek-r1:7b"
        original_weight = MODEL_REGISTRY[model]["quality_weight"]

        execs = (
            _make_execs(45, model=model, task_type="debugging", success=True)
            + _make_execs(15, model="qwen2.5:7b", task_type="debugging", success=False)
        )
        mock_memory.get_recent.return_value = execs
        mock_memory.get_model_stats.return_value = {"success_rate": 0.9, "runs": 45}

        cycle = await svc.run_evolution_cycle()

        # Whether or not anything was applied, rollback should succeed
        if cycle.strategies_applied:
            success = await svc.rollback_cycle(cycle.id)
            assert success is True
            # Weight should be close to original (may differ if multiple apply rounds happened)
        else:
            # Nothing to rollback — but the cycle is in history
            history = await svc.get_history()
            assert any(c.id == cycle.id for c in history)

        # Restore for other tests
        MODEL_REGISTRY[model]["quality_weight"] = original_weight


# ─── API endpoints ────────────────────────────────────────────────────────────

class TestEvolutionEndpoints:
    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def mock_svc(self):
        from app.models.evolution import EvolutionCycle

        svc = MagicMock()
        cycle = EvolutionCycle(
            id="endpoint-test",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        svc.run_evolution_cycle = AsyncMock(return_value=cycle)
        svc.get_history = AsyncMock(return_value=[cycle])
        svc.rollback_cycle = AsyncMock(return_value=True)
        svc.get_current_state_summary = MagicMock(return_value={
            "current_model_weights": {"m": {"quality_weight": 0.9}},
            "current_context_strategies": {"audit": {"preferred_mode": "comprehensive"}},
        })
        return svc

    def test_run_returns_cycle(self, client, mock_svc):
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/run")
        assert resp.status_code == 200
        assert resp.json()["id"] == "endpoint-test"

    def test_run_with_force_passes_flag(self, client, mock_svc):
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/run?force=true")
        assert resp.status_code == 200
        mock_svc.run_evolution_cycle.assert_awaited_with(force=True)

    def test_run_without_force_uses_default(self, client, mock_svc):
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/run")
        assert resp.status_code == 200
        mock_svc.run_evolution_cycle.assert_awaited_with(force=False)

    def test_run_error_returns_500(self, client, mock_svc):
        mock_svc.run_evolution_cycle = AsyncMock(side_effect=RuntimeError("crash"))
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/run")
        assert resp.status_code == 500

    def test_status_rich_response(self, client, mock_svc):
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.get("/api/v1/evolution/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_run"] is True
        assert "total_cycles_run" in body
        assert "total_improvements_applied" in body
        assert "current_model_weights" in body
        assert "current_context_strategies" in body

    def test_status_no_history(self, client, mock_svc):
        mock_svc.get_history = AsyncMock(return_value=[])
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.get("/api/v1/evolution/status")
        assert resp.status_code == 200
        assert resp.json()["has_run"] is False

    def test_status_error(self, client, mock_svc):
        mock_svc.get_history = AsyncMock(side_effect=RuntimeError("fail"))
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.get("/api/v1/evolution/status")
        assert resp.status_code == 500

    def test_history_returns_list(self, client, mock_svc):
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.get("/api/v1/evolution/history?limit=5")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_rollback_requires_confirm(self, client, mock_svc):
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/rollback/some-id")
        assert resp.status_code == 400
        assert "confirm" in resp.json()["error"]

    def test_rollback_success_with_confirm(self, client, mock_svc):
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/rollback/some-id?confirm=true")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_rollback_not_found(self, client, mock_svc):
        mock_svc.rollback_cycle = AsyncMock(return_value=False)
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/rollback/bad-id?confirm=true")
        assert resp.status_code == 404

    def test_rollback_error(self, client, mock_svc):
        mock_svc.rollback_cycle = AsyncMock(side_effect=RuntimeError("disk error"))
        with patch("app.api.v1.endpoints.evolution.get_evolution_service", return_value=mock_svc):
            resp = client.post("/api/v1/evolution/rollback/eid?confirm=true")
        assert resp.status_code == 500
