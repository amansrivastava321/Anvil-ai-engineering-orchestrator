"""Evolution Service — closes the learning loop.

Analyzes execution history, discovers patterns, validates improvements,
and applies strategy changes to running services automatically.

Design principles:
- Every analysis method reads real data (no placeholders)
- Strategies are validated against historical data before application
- All changes are reversible via previous_state / rollback_instructions
- Never raises from run_evolution_cycle — all errors captured in the cycle record
- Concurrent cycle prevention: only one cycle can run at a time
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.monitoring.logging import get_logger
from app.memory.execution_memory import ExecutionMemory
from app.memory.pattern_store import PatternStore
from app.models.evolution import (
    EvolutionCycle,
    EvolutionRecommendation,
    ModelPerformanceRecord,
    RiskLevel,
    StrategyType,
    StrategyUpdate,
    WorkflowPerformanceRecord,
)

logger = get_logger(__name__)

__all__ = ["EvolutionService", "get_evolution_service"]

_MIN_EXECUTIONS = 50
_IMPROVEMENT_THRESHOLD = 0.05           # 5 % minimum gain to apply
_EXECUTIONS_DIR = Path("data/performance_db/executions")
_HISTORY_PATH = Path("data/performance_db/strategies/evolution_history.json")


class EvolutionService:
    """Self-improvement engine for the Engineering Intelligence OS.

    Usage::

        svc = EvolutionService()
        cycle = await svc.run_evolution_cycle()
    """

    def __init__(
        self,
        execution_memory: Optional[ExecutionMemory] = None,
        pattern_store: Optional[PatternStore] = None,
        improvement_threshold: float = _IMPROVEMENT_THRESHOLD,
        min_executions: int = _MIN_EXECUTIONS,
        max_automatic_risk: str = "medium",
        auto_apply: bool = False,
    ) -> None:
        from app.memory.execution_memory import ExecutionMemory as EM
        from app.memory.pattern_store import PatternStore as PS

        self._memory = execution_memory or EM()
        self._patterns = pattern_store or PS()
        self._improvement_threshold = improvement_threshold
        self._min_executions = min_executions
        self._max_automatic_risk = max_automatic_risk
        self._auto_apply = auto_apply
        self._current_cycle: Optional[EvolutionCycle] = None
        self._running = False  # concurrent-cycle guard
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run_evolution_cycle(self, force: bool = False) -> EvolutionCycle:
        """Run a complete evolution cycle and return the results.

        Args:
            force: Skip the minimum-executions guard and run even with sparse data.
        """
        if self._running:
            logger.warning("Evolution cycle already running — skipping concurrent request")
            placeholder = EvolutionCycle(
                id=str(uuid.uuid4()),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                error="Concurrent evolution cycle already running",
            )
            return placeholder

        self._running = True
        cycle = EvolutionCycle(
            id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
        )
        self._current_cycle = cycle
        logger.info("Evolution cycle started", cycle_id=cycle.id, force=force)

        try:
            executions = await self._load_execution_data()
            cycle.executions_analyzed = len(executions)

            if not force and len(executions) < self._min_executions:
                logger.info(
                    "Insufficient data for evolution",
                    have=len(executions),
                    need=self._min_executions,
                )
                cycle.completed_at = datetime.now(timezone.utc)
                cycle.improvements_achieved = {
                    "skipped": (
                        f"Insufficient data: need {self._min_executions} executions, "
                        f"have {len(executions)}"
                    )
                }
                await self._record_evolution_cycle(cycle)
                return cycle

            # ── Analysis ──────────────────────────────────────────────────────
            model_perf = await self._analyze_model_performance(executions)
            wf_perf = await self._analyze_workflow_performance(executions)
            skill_data = await self._analyze_skill_combinations()
            ctx_data = await self._analyze_context_strategies(executions)

            cycle.model_performance = model_perf
            cycle.workflow_performance = wf_perf

            all_findings = {
                "model_performance": {
                    r.task_type: {
                        "best_model": next(
                            (x.model_name for x in model_perf if x.task_type == r.task_type and x.is_best),
                            None,
                        ),
                        "success_rate": r.success_rate,
                        "runs": r.executions,
                        "confidence": min(1.0, r.executions / 100),
                        "all_models": [
                            {
                                "model": x.model_name,
                                "success_rate": x.success_rate,
                                "runs": x.executions,
                                "avg_duration_ms": x.avg_duration_ms,
                            }
                            for x in model_perf
                            if x.task_type == r.task_type
                        ],
                    }
                    for r in model_perf
                    if r.is_best
                },
                "workflow_effectiveness": {
                    r.workflow_name: {
                        "runs": r.executions,
                        "success_rate": r.completion_rate,
                        "error_rate": r.error_rate,
                        "needs_attention": r.completion_rate < 0.70 and r.executions >= 5,
                    }
                    for r in wf_perf
                },
                "skill_combinations": skill_data,
                "context_strategies": ctx_data,
            }

            for category, findings in all_findings.items():
                if findings:
                    cycle.patterns_discovered.append(
                        {"category": category, "findings": findings}
                    )

            # ── Generate → validate → apply ───────────────────────────────────
            updates = await self._generate_strategies(all_findings)
            cycle.strategies_generated = updates

            validated = await self._validate_strategies(updates, executions)
            cycle.strategies_validated = validated

            appliable = [
                u for u in validated
                if u.is_auto_appliable(self._max_automatic_risk)
            ]
            applied = await self._apply_strategies(appliable)
            cycle.strategies_applied = applied

            # ── Recommendations ───────────────────────────────────────────────
            cycle.recommendations = self._build_recommendations(applied, all_findings)

            # ── Improvements summary ──────────────────────────────────────────
            cycle.improvements_achieved = self._calculate_improvements(applied)

        except Exception as exc:
            cycle.error = str(exc)
            logger.error("Evolution cycle failed", cycle_id=cycle.id, error=str(exc))

        finally:
            self._running = False

        cycle.completed_at = datetime.now(timezone.utc)
        await self._record_evolution_cycle(cycle)
        logger.info(
            "Evolution cycle complete",
            cycle_id=cycle.id,
            applied=len(cycle.strategies_applied),
            duration_s=cycle.duration_seconds,
        )
        return cycle

    async def schedule_evolution_cycles(self, interval_hours: int = 168) -> None:
        """Background loop: runs an evolution cycle every ``interval_hours`` hours."""
        interval = interval_hours * 3600
        logger.info("Evolution scheduler started", interval_hours=interval_hours)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.run_evolution_cycle()
            except asyncio.CancelledError:
                logger.info("Evolution scheduler stopped")
                return
            except Exception as exc:
                logger.error("Scheduled evolution cycle failed", error=str(exc))

    # ── Data loading ───────────────────────────────────────────────────────────

    async def _load_execution_data(self) -> List[Dict[str, Any]]:
        """Read all execution JSON files from ``data/performance_db/executions/``.

        Merges with ExecutionMemory so no records are missed.
        Corrupted or unreadable files are skipped with a warning.
        """
        file_executions: List[Dict[str, Any]] = []

        if _EXECUTIONS_DIR.exists():
            for json_file in sorted(_EXECUTIONS_DIR.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text())
                    if isinstance(data, dict):
                        file_executions.append(data)
                    elif isinstance(data, list):
                        file_executions.extend(data)
                except Exception as exc:
                    logger.warning(
                        "Skipping corrupted execution file",
                        file=str(json_file),
                        error=str(exc),
                    )

        # Merge in-memory records (deduplication by run_id)
        seen_ids = {ex.get("run_id") for ex in file_executions}
        for mem_ex in self._memory.get_recent(limit=1000):
            if mem_ex.get("run_id") not in seen_ids:
                file_executions.append(mem_ex)
                seen_ids.add(mem_ex.get("run_id"))

        return file_executions

    # ── Analysis ───────────────────────────────────────────────────────────────

    async def _analyze_model_performance(
        self, executions: List[Dict[str, Any]]
    ) -> List[ModelPerformanceRecord]:
        """Rank models by success rate, speed, and efficiency for each task type."""
        by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for ex in executions:
            task = ex.get("task_type", "unknown")
            model = ex.get("model", "unknown")
            success = bool(ex.get("success", False))

            task_data = by_task.setdefault(task, {})
            if model not in task_data:
                task_data[model] = {
                    "runs": 0,
                    "successes": 0,
                    "total_duration_ms": 0.0,
                    "total_tokens": 0,
                }
            task_data[model]["runs"] += 1
            if success:
                task_data[model]["successes"] += 1
            task_data[model]["total_duration_ms"] += ex.get("duration_ms", 0.0)
            task_data[model]["total_tokens"] += (
                ex.get("metadata", {}).get("tokens_used", 0) if isinstance(ex.get("metadata"), dict) else 0
            )

        records: List[ModelPerformanceRecord] = []
        for task, model_data in by_task.items():
            ranked = []
            for model, stats in model_data.items():
                runs = stats["runs"]
                sr = stats["successes"] / runs if runs else 0.0
                ranked.append(
                    (
                        model,
                        runs,
                        sr,
                        stats["total_duration_ms"] / runs if runs else 0.0,
                        stats["total_tokens"] / runs if runs else 0.0,
                    )
                )
            ranked.sort(key=lambda x: (-x[2], x[3]))  # best success rate, then fastest

            for rank, (model, runs, sr, dur, tok) in enumerate(ranked, start=1):
                records.append(
                    ModelPerformanceRecord(
                        model_name=model,
                        task_type=task,
                        executions=runs,
                        success_rate=round(sr, 4),
                        avg_duration_ms=round(dur, 1),
                        avg_tokens_used=round(tok, 1),
                        rank=rank,
                        is_best=(rank == 1),
                    )
                )

        return records

    async def _analyze_workflow_performance(
        self, executions: List[Dict[str, Any]]
    ) -> List[WorkflowPerformanceRecord]:
        """Compute completion rate and error rate per workflow type."""
        by_wf: Dict[str, Dict[str, Any]] = {}
        for ex in executions:
            wf = ex.get("workflow", "unknown")
            success = bool(ex.get("success", False))
            has_error = bool(ex.get("error"))

            if wf not in by_wf:
                by_wf[wf] = {"runs": 0, "successes": 0, "errors": 0}
            by_wf[wf]["runs"] += 1
            if success:
                by_wf[wf]["successes"] += 1
            if has_error:
                by_wf[wf]["errors"] += 1

        records: List[WorkflowPerformanceRecord] = []
        for wf, stats in by_wf.items():
            runs = stats["runs"]
            cr = stats["successes"] / runs if runs else 0.0
            er = stats["errors"] / runs if runs else 0.0
            effectiveness = max(0.0, cr - er * 0.5)
            records.append(
                WorkflowPerformanceRecord(
                    workflow_name=wf,
                    executions=runs,
                    completion_rate=round(cr, 4),
                    error_rate=round(er, 4),
                    effectiveness_score=round(effectiveness, 4),
                )
            )

        return sorted(records, key=lambda r: -r.effectiveness_score)

    async def _analyze_skill_combinations(self) -> Dict[str, Any]:
        """Discover skill combinations that correlate with successful outcomes."""
        patterns = self._patterns.get_patterns("skill_combination")
        if not patterns:
            return {}

        combos: Dict[str, Dict[str, Any]] = {}
        for pattern in patterns:
            key = pattern.get("key", "")
            outcomes = pattern.get("outcomes", [])
            if not outcomes:
                continue
            successes = sum(
                1 for o in outcomes
                if o.get("outcome", "") in ("success", "passed", "completed")
            )
            total = len(outcomes)
            combos[key] = {
                "count": total,
                "success_rate": round(successes / total, 4) if total else 0.0,
                "first_seen": pattern.get("first_seen"),
                "last_seen": pattern.get("last_seen"),
            }

        single_rates = {k: v["success_rate"] for k, v in combos.items() if "+" not in k}
        recommended = []
        for key, stats in combos.items():
            if "+" not in key:
                continue
            baseline = max(
                (single_rates.get(p.strip(), 0.0) for p in key.split("+")), default=0.0
            )
            lift = stats["success_rate"] - baseline
            if lift >= 0.10:
                recommended.append(
                    {"combination": key, "lift_over_baseline": round(lift, 4), **stats}
                )

        return {
            "total_combinations": len(combos),
            "combinations": combos,
            "recommended": sorted(recommended, key=lambda x: -x["lift_over_baseline"]),
        }

    async def _analyze_context_strategies(
        self, executions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find which context modes correlate with better outcomes per workflow."""
        by_wf_mode: Dict[str, Dict[str, Dict[str, int]]] = {}
        for ex in executions:
            wf = ex.get("workflow", "unknown")
            mode = (
                ex.get("metadata", {}).get("context_mode", "balanced")
                if isinstance(ex.get("metadata"), dict)
                else "balanced"
            )
            success = bool(ex.get("success", False))
            wf_data = by_wf_mode.setdefault(wf, {})
            if mode not in wf_data:
                wf_data[mode] = {"runs": 0, "successes": 0}
            wf_data[mode]["runs"] += 1
            if success:
                wf_data[mode]["successes"] += 1

        result: Dict[str, Any] = {}
        for wf, mode_data in by_wf_mode.items():
            ranked = []
            for mode, stats in mode_data.items():
                runs = stats["runs"]
                sr = stats["successes"] / runs if runs else 0.0
                ranked.append({"mode": mode, "runs": runs, "success_rate": round(sr, 4)})
            ranked.sort(key=lambda x: -x["success_rate"])
            if ranked:
                result[wf] = {
                    "best_mode": ranked[0]["mode"],
                    "best_success_rate": ranked[0]["success_rate"],
                    "all_modes": ranked,
                }

        return result

    # ── Strategy lifecycle ─────────────────────────────────────────────────────

    async def _generate_strategies(
        self, analysis: Dict[str, Any]
    ) -> List[StrategyUpdate]:
        """Convert analysis findings into concrete StrategyUpdate proposals."""
        updates: List[StrategyUpdate] = []

        # MODEL_SELECTION
        for task_type, data in analysis.get("model_performance", {}).items():
            all_models = data.get("all_models", [])
            if len(all_models) < 2:
                continue
            best = all_models[0]
            second = all_models[1]
            improvement = best["success_rate"] - second["success_rate"]
            if improvement < self._improvement_threshold:
                continue
            if data.get("confidence", 0) < 0.3:
                continue
            updates.append(
                StrategyUpdate(
                    id=str(uuid.uuid4()),
                    strategy_type=StrategyType.MODEL_SELECTION,
                    current_behavior=(
                        f"Task '{task_type}': {second['model']} "
                        f"({second['success_rate']:.1%} success)"
                    ),
                    proposed_behavior=(
                        f"Task '{task_type}': prefer {best['model']} "
                        f"({best['success_rate']:.1%} success)"
                    ),
                    evidence={
                        "task_type": task_type,
                        "best_model": best["model"],
                        "best_success_rate": best["success_rate"],
                        "confidence": data.get("confidence", 0),
                        "sample_size": best["runs"],
                        "improvement_over_second": round(improvement, 4),
                    },
                    expected_improvement_percent=round(improvement * 100, 2),
                    risk_level=RiskLevel.LOW if improvement > 0.15 else RiskLevel.MEDIUM,
                )
            )

        # WORKFLOW_DESIGN
        for wf, data in analysis.get("workflow_effectiveness", {}).items():
            if not data.get("needs_attention"):
                continue
            updates.append(
                StrategyUpdate(
                    id=str(uuid.uuid4()),
                    strategy_type=StrategyType.WORKFLOW_DESIGN,
                    current_behavior=(
                        f"Workflow '{wf}': {data['success_rate']:.1%} success rate"
                    ),
                    proposed_behavior=(
                        f"Workflow '{wf}': add pre-validation + increase context budget"
                    ),
                    evidence={
                        "workflow": wf,
                        "success_rate": data["success_rate"],
                        "error_rate": data["error_rate"],
                        "runs": data["runs"],
                    },
                    expected_improvement_percent=10.0,
                    risk_level=RiskLevel.MEDIUM,
                )
            )

        # SKILL_INJECTION — top 3 high-lift combos
        for combo in analysis.get("skill_combinations", {}).get("recommended", [])[:3]:
            lift_pct = round(combo["lift_over_baseline"] * 100, 2)
            updates.append(
                StrategyUpdate(
                    id=str(uuid.uuid4()),
                    strategy_type=StrategyType.SKILL_INJECTION,
                    current_behavior="Skills injected individually",
                    proposed_behavior=f"Inject combo: {combo['combination']}",
                    evidence={
                        "combination": combo["combination"],
                        "success_rate": combo["success_rate"],
                        "lift_over_baseline": combo["lift_over_baseline"],
                        "sample_count": combo["count"],
                    },
                    expected_improvement_percent=lift_pct,
                    risk_level=RiskLevel.LOW,
                )
            )

        # CONTEXT_ASSEMBLY
        for wf, data in analysis.get("context_strategies", {}).items():
            all_modes = data.get("all_modes", [])
            if len(all_modes) < 2:
                continue
            best = all_modes[0]
            worst = all_modes[-1]
            improvement = best["success_rate"] - worst["success_rate"]
            if improvement < self._improvement_threshold:
                continue
            updates.append(
                StrategyUpdate(
                    id=str(uuid.uuid4()),
                    strategy_type=StrategyType.CONTEXT_ASSEMBLY,
                    current_behavior=(
                        f"Workflow '{wf}': mode '{worst['mode']}' "
                        f"({worst['success_rate']:.1%} success)"
                    ),
                    proposed_behavior=(
                        f"Workflow '{wf}': mode '{best['mode']}' "
                        f"({best['success_rate']:.1%} success)"
                    ),
                    evidence={
                        "workflow": wf,
                        "best_mode": best["mode"],
                        "improvement": round(improvement, 4),
                        "sample_size": best["runs"],
                    },
                    expected_improvement_percent=round(improvement * 100, 2),
                    risk_level=RiskLevel.LOW if improvement > 0.15 else RiskLevel.MEDIUM,
                )
            )

        logger.info("Generated strategy updates", count=len(updates))
        return updates

    async def _validate_strategies(
        self, updates: List[StrategyUpdate], executions: List[Dict[str, Any]]
    ) -> List[StrategyUpdate]:
        """Validate a list of strategy updates against historical data.

        Populates ``validation_result`` on each update and returns only those
        that passed.
        """
        passed: List[StrategyUpdate] = []
        for update in updates:
            result = await self._validate_single(update, executions)
            update.validation_result = result
            if result:
                passed.append(update)
        return passed

    async def _validate_single(
        self, update: StrategyUpdate, executions: List[Dict[str, Any]]
    ) -> bool:
        """Simulate the proposed strategy on recent executions.

        Evidence-only strategies (SKILL_INJECTION, CONTEXT_ASSEMBLY,
        WORKFLOW_DESIGN) do not require execution samples.
        """
        if update.strategy_type == StrategyType.SKILL_INJECTION:
            return update.evidence.get("lift_over_baseline", 0.0) >= self._improvement_threshold

        if update.strategy_type == StrategyType.CONTEXT_ASSEMBLY:
            return update.evidence.get("improvement", 0.0) >= self._improvement_threshold

        if update.strategy_type == StrategyType.WORKFLOW_DESIGN:
            sr = update.evidence.get("success_rate", 1.0)
            runs = update.evidence.get("runs", 0)
            return sr < 0.70 and runs >= 5

        # MODEL_SELECTION — simulate against sample
        sample = executions[-10:] if len(executions) >= 10 else executions
        if update.strategy_type == StrategyType.MODEL_SELECTION:
            if not sample:
                # Fall back to evidence-based confidence check
                return (
                    update.evidence.get("confidence", 0) >= 0.5
                    and update.expected_improvement_percent / 100 >= self._improvement_threshold
                )
            proposed_model = update.evidence.get("best_model", "")
            task_type = update.evidence.get("task_type", "")
            relevant = [ex for ex in sample if ex.get("task_type") == task_type]
            if not relevant:
                return (
                    update.evidence.get("confidence", 0) >= 0.5
                    and update.expected_improvement_percent / 100 >= self._improvement_threshold
                )
            proposed_stats = self._memory.get_model_stats(proposed_model)
            proposed_sr = proposed_stats.get("success_rate", 0.0)
            actual_sr = sum(1 for ex in relevant if ex.get("success")) / len(relevant)
            return proposed_sr - actual_sr >= self._improvement_threshold

        return update.expected_improvement_percent / 100 >= self._improvement_threshold

    async def _apply_strategies(
        self, updates: List[StrategyUpdate]
    ) -> List[StrategyUpdate]:
        """Apply validated strategy updates. Returns the subset that succeeded."""
        applied: List[StrategyUpdate] = []
        for update in updates:
            try:
                rollback = await self._apply_single(update)
                update.applied = True
                update.applied_at = datetime.now(timezone.utc)
                update.rollback_instructions = json.dumps(rollback)
                update.previous_state = rollback
                applied.append(update)
                logger.info(
                    "Strategy applied",
                    strategy_id=update.id,
                    type=update.strategy_type.value,
                )
            except Exception as exc:
                logger.error(
                    "Failed to apply strategy",
                    strategy_id=update.id,
                    error=str(exc),
                )
        return applied

    async def _apply_single(self, update: StrategyUpdate) -> Dict[str, Any]:
        """Apply one strategy update and return rollback instructions."""
        if update.strategy_type == StrategyType.MODEL_SELECTION:
            return await self._apply_model_selection(update)
        elif update.strategy_type == StrategyType.CONTEXT_ASSEMBLY:
            return await self._apply_context_assembly(update)
        elif update.strategy_type == StrategyType.SKILL_INJECTION:
            return await self._apply_skill_injection(update)
        elif update.strategy_type == StrategyType.WORKFLOW_DESIGN:
            return await self._apply_workflow_design(update)
        else:
            return {"note": f"{update.strategy_type.value} — recorded, manual review needed"}

    async def _apply_model_selection(self, update: StrategyUpdate) -> Dict[str, Any]:
        from app.services.model_service import get_model_service

        svc = get_model_service()
        previous_weights = svc.get_current_weights()
        best_model = update.evidence.get("best_model", "")
        best_sr = update.evidence.get("best_success_rate", 0.8)
        new_weight = min(1.0, best_sr + 0.05)
        svc.update_model_weights({best_model: new_weight})
        return {
            "type": "model_selection",
            "previous_weights": previous_weights,
            "model_updated": best_model,
            "old_weight": previous_weights.get(best_model, {}).get("quality_weight"),
            "new_weight": new_weight,
        }

    async def _apply_context_assembly(self, update: StrategyUpdate) -> Dict[str, Any]:
        from app.services.context_service import get_context_service

        svc = get_context_service()
        wf = update.evidence.get("workflow", "")
        best_mode = update.evidence.get("best_mode", "balanced")
        previous = svc.get_context_strategy(wf)
        svc.update_context_strategy(wf, {"preferred_mode": best_mode})
        return {
            "type": "context_assembly",
            "workflow": wf,
            "previous_strategy": previous,
            "new_strategy": {"preferred_mode": best_mode},
        }

    async def _apply_skill_injection(self, update: StrategyUpdate) -> Dict[str, Any]:
        from app.services.context_service import get_context_service

        svc = get_context_service()
        combination = update.evidence.get("combination", "")
        previous_max = svc._max_skills_per_task
        svc.set_skill_limits(max_skills=max(previous_max, 4))
        combos = getattr(svc, "_preferred_skill_combinations", [])
        if combination not in combos:
            combos.append(combination)
        svc._preferred_skill_combinations = combos
        return {
            "type": "skill_injection",
            "combination_added": combination,
            "previous_max_skills": previous_max,
            "new_max_skills": svc._max_skills_per_task,
        }

    async def _apply_workflow_design(self, update: StrategyUpdate) -> Dict[str, Any]:
        self._patterns.record_pattern(
            pattern_type="workflow_improvement",
            pattern_key=update.evidence.get("workflow", "unknown"),
            context=update.evidence,
            outcome="flagged_for_review",
        )
        return {
            "type": "workflow_design",
            "note": "Recorded as pattern — requires human review",
            "workflow": update.evidence.get("workflow"),
        }

    # ── Improvements summary ───────────────────────────────────────────────────

    def _calculate_improvements(
        self, applied: List[StrategyUpdate]
    ) -> Dict[str, Any]:
        """Summarize expected gains from all applied strategies."""
        if not applied:
            return {"total_strategies_applied": 0, "expected_improvements": []}

        by_type: Dict[str, List[float]] = {}
        total_expected = 0.0
        for u in applied:
            t = u.strategy_type.value
            by_type.setdefault(t, []).append(u.expected_improvement_percent)
            total_expected += u.expected_improvement_percent

        return {
            "total_strategies_applied": len(applied),
            "total_expected_improvement_percent": round(total_expected, 2),
            "strategies_applied": len(applied),
            "strategies_skipped_risk": 0,
            "strategies_failed_validation": 0,
            "by_type": {
                t: {
                    "count": len(vals),
                    "avg_expected_improvement_percent": round(sum(vals) / len(vals), 2),
                }
                for t, vals in by_type.items()
            },
            "expected_improvements": [
                {
                    "strategy": u.strategy_type.value,
                    "expected_percent": u.expected_improvement_percent,
                    "risk": u.risk_level.value,
                }
                for u in applied
            ],
        }

    # ── Recommendations ────────────────────────────────────────────────────────

    def _build_recommendations(
        self,
        applied: List[StrategyUpdate],
        findings: Dict[str, Any],
    ) -> List[EvolutionRecommendation]:
        """Build human-readable recommendations from applied updates."""
        recs: List[EvolutionRecommendation] = []
        if not applied:
            return recs

        # Group by strategy type
        by_type: Dict[str, List[StrategyUpdate]] = {}
        for u in applied:
            by_type.setdefault(u.strategy_type.value, []).append(u)

        for strategy_type, updates in by_type.items():
            total_improvement = sum(u.expected_improvement_percent for u in updates)
            recs.append(
                EvolutionRecommendation(
                    title=f"{strategy_type.replace('_', ' ').title()} Optimized",
                    description=(
                        f"Applied {len(updates)} {strategy_type} change(s) "
                        f"with expected {total_improvement:.1f}% improvement."
                    ),
                    strategy_updates=updates,
                    expected_cumulative_improvement_percent=round(total_improvement, 2),
                    priority=1 if total_improvement > 20 else 2,
                    auto_apply=True,
                )
            )

        return sorted(recs, key=lambda r: r.priority)

    # ── History & rollback ─────────────────────────────────────────────────────

    async def get_history(self, limit: int = 10) -> List[EvolutionCycle]:
        """Load evolution cycle history from disk (newest first)."""
        if not _HISTORY_PATH.exists():
            return []
        try:
            raw = json.loads(_HISTORY_PATH.read_text())
            cycles = [EvolutionCycle.model_validate(entry) for entry in raw]
            return cycles[-limit:][::-1]
        except Exception as exc:
            logger.error("Failed to read evolution history", error=str(exc))
            return []

    async def rollback_cycle(self, cycle_id: str) -> bool:
        """Reverse all strategy changes applied during a given cycle."""
        history = await self.get_history(limit=100)
        target = next((c for c in history if c.id == cycle_id), None)
        if target is None:
            logger.warning("Rollback target not found", cycle_id=cycle_id)
            return False

        rolled_back = 0
        for update in target.strategies_applied:
            instructions = update.rollback_instructions or (
                json.dumps(update.previous_state) if update.previous_state else None
            )
            if not instructions:
                continue
            try:
                await self._execute_rollback(json.loads(instructions))
                rolled_back += 1
            except Exception as exc:
                logger.error(
                    "Rollback failed for strategy",
                    strategy_id=update.id,
                    error=str(exc),
                )

        target.rollback_performed = True
        await self._update_cycle_in_history(target)
        logger.info(
            "Rollback complete", cycle_id=cycle_id, strategies_reversed=rolled_back
        )
        return True

    async def _execute_rollback(self, instructions: Dict[str, Any]) -> None:
        rt = instructions.get("type")
        if rt == "model_selection":
            from app.services.model_service import get_model_service

            svc = get_model_service()
            prev = instructions.get("previous_weights", {})
            restore = {
                model: data.get("quality_weight", 0.5)
                for model, data in prev.items()
            }
            if restore:
                svc.update_model_weights(restore)

        elif rt == "context_assembly":
            from app.services.context_service import get_context_service

            svc = get_context_service()
            wf = instructions.get("workflow", "")
            prev = instructions.get("previous_strategy", {})
            svc.update_context_strategy(wf, prev)

        elif rt == "skill_injection":
            from app.services.context_service import get_context_service

            svc = get_context_service()
            prev_max = instructions.get("previous_max_skills", 3)
            svc.set_skill_limits(max_skills=prev_max)

    async def record_evolution(self, cycle: EvolutionCycle) -> None:
        """Public alias used by external callers and tests."""
        await self._record_evolution_cycle(cycle)

    # ── Persistence ────────────────────────────────────────────────────────────

    async def _record_evolution_cycle(self, cycle: EvolutionCycle) -> None:
        try:
            existing: List[Dict[str, Any]] = []
            if _HISTORY_PATH.exists():
                existing = json.loads(_HISTORY_PATH.read_text())
            existing.append(cycle.model_dump(mode="json"))
            _HISTORY_PATH.write_text(json.dumps(existing, indent=2, default=str))
        except Exception as exc:
            logger.error("Failed to save evolution cycle", error=str(exc))

    async def _update_cycle_in_history(self, cycle: EvolutionCycle) -> None:
        try:
            if not _HISTORY_PATH.exists():
                return
            existing: List[Dict[str, Any]] = json.loads(_HISTORY_PATH.read_text())
            for i, entry in enumerate(existing):
                if entry.get("id") == cycle.id:
                    existing[i] = cycle.model_dump(mode="json")
                    break
            _HISTORY_PATH.write_text(json.dumps(existing, indent=2, default=str))
        except Exception as exc:
            logger.error("Failed to update cycle in history", error=str(exc))

    def get_current_state_summary(self) -> Dict[str, Any]:
        """Snapshot of live model weights and context strategies (for /status)."""
        try:
            from app.services.model_service import get_model_service
            weights = get_model_service().get_current_weights()
        except Exception:
            weights = {}

        try:
            from app.services.context_service import get_context_service
            ctx_strategies = get_context_service()._context_strategies
        except Exception:
            ctx_strategies = {}

        return {
            "current_model_weights": weights,
            "current_context_strategies": ctx_strategies,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_evolution_service: Optional[EvolutionService] = None


def get_evolution_service() -> EvolutionService:
    """Return the application-wide EvolutionService singleton."""
    global _evolution_service
    if _evolution_service is None:
        _evolution_service = EvolutionService()
    return _evolution_service
