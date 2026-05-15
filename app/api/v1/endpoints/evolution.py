"""Evolution API endpoints — trigger, inspect, and rollback evolution cycles."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from app.core.monitoring.logging import get_logger
from app.services.evolution_service import get_evolution_service

logger = get_logger(__name__)
router = APIRouter()


@router.post("/run", summary="Trigger an evolution cycle manually")
async def trigger_evolution_cycle(
    force: bool = Query(
        default=False,
        description="Skip the minimum-executions guard and run even with sparse data",
    ),
) -> Dict[str, Any]:
    """Run a full evolution cycle immediately and return the results.

    Pass ``force=true`` to bypass the minimum-executions check — useful for
    demos and debugging with small datasets.
    """
    svc = get_evolution_service()
    try:
        cycle = await svc.run_evolution_cycle(force=force)
        return cycle.model_dump(mode="json")
    except Exception as exc:
        logger.error("Evolution cycle failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status", summary="Get current evolution state")
async def evolution_status() -> Dict[str, Any]:
    """Return a rich summary of the current evolution state.

    Includes:
    - Total cycles run and improvements applied
    - Current live model weights (shows what evolution has changed)
    - Current live context strategies per workflow
    - Most recent cycle summary
    """
    svc = get_evolution_service()
    try:
        history = await svc.get_history(limit=100)
        last_cycle = history[0].model_dump(mode="json") if history else None
        total_applied = sum(len(c.strategies_applied) for c in history)
        state = svc.get_current_state_summary()

        return {
            "has_run": last_cycle is not None,
            "total_cycles_run": len(history),
            "total_improvements_applied": total_applied,
            "last_cycle": last_cycle,
            "current_model_weights": state["current_model_weights"],
            "current_context_strategies": state["current_context_strategies"],
        }
    except Exception as exc:
        logger.error("Failed to get evolution status", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history", summary="List evolution cycle history")
async def evolution_history(
    limit: int = Query(default=10, ge=1, le=100),
) -> List[Dict[str, Any]]:
    """Return the last N evolution cycles, newest first."""
    svc = get_evolution_service()
    try:
        cycles = await svc.get_history(limit=limit)
        return [c.model_dump(mode="json") for c in cycles]
    except Exception as exc:
        logger.error("Failed to fetch evolution history", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/rollback/{cycle_id}", summary="Rollback a specific evolution cycle")
async def rollback_evolution(
    cycle_id: str,
    confirm: bool = Query(
        default=False,
        description="Must be true to actually execute the rollback",
    ),
) -> Dict[str, Any]:
    """Reverse all strategy changes applied during the given cycle.

    Requires ``confirm=true`` to prevent accidental rollbacks.
    Returns 404 if the cycle ID is not found in history.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Pass confirm=true to execute the rollback",
        )

    svc = get_evolution_service()
    try:
        success = await svc.rollback_cycle(cycle_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Evolution cycle '{cycle_id}' not found in history",
            )
        return {"success": True, "cycle_id": cycle_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Rollback failed", cycle_id=cycle_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
