"""Agent API endpoints — all requests go through the CEO AI."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.ai.ceo import get_ceo
from app.services.orchestrator import (
    OrchestratorRequest,
    get_orchestrator,
)
from app.core.monitoring.logging import get_logger
from app.utils.validators import PathSecurityError

logger = get_logger(__name__)

router = APIRouter()


@router.post("/run")
async def run_agent(request: OrchestratorRequest):
    """Execute an AI engineering workflow through the CEO.

    Every request goes to the CEO first. The CEO selects its operating mode
    (decide alone, consult experts, or convene council) and returns a decision
    record with full reasoning and attribution.

    Falls back to the legacy orchestrator if the CEO is unavailable.
    """
    ceo = get_ceo()
    try:
        decision = await ceo.receive_problem(
            description=request.prompt,
            context={
                "repo_path": request.repo_path,
                "workflow_type": request.workflow_type.value,
                "include_files": request.include_files or [],
            },
            risk_level="medium",
        )
        return {
            "execution_id": decision.id,
            "status": "completed",
            "workflow_type": request.workflow_type,
            "model_used": "ceo-council",
            "response": decision.final_plan,
            "ceo_mode": decision.mode.value,
            "experts_consulted": decision.experts_consulted,
            "reasoning": decision.mode_reasoning,
            "confidence": decision.confidence,
            "patterns_matched": decision.patterns_matched,
            "duration_ms": decision.duration_ms,
            "artifacts": [],
            "warnings": [],
        }
    except Exception as exc:
        logger.warning("CEO unavailable, falling back to orchestrator", error=str(exc))
        orchestrator = get_orchestrator()
        try:
            response = await orchestrator.execute(request)
            return response.model_dump()
        except (ValueError, PathSecurityError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Agent execution failed", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/stream")
async def run_agent_streaming(request: OrchestratorRequest):
    """Execute workflow with streaming response via the legacy orchestrator."""
    orchestrator = get_orchestrator()

    async def generate():
        try:
            async for token in orchestrator.execute_streaming(request):
                yield token
        except ValueError as e:
            yield f"\n[Error 400] {e}"
        except Exception as e:
            logger.error("Streaming agent failed", error=str(e), exc_info=True)
            yield "\n[Error 500] Internal server error"

    return StreamingResponse(generate(), media_type="text/plain")


@router.get("/executions")
async def get_active_executions():
    """Get currently active executions."""
    orchestrator = get_orchestrator()
    return {"executions": orchestrator.get_active_executions()}


@router.get("/stats")
async def get_execution_stats():
    """Get execution statistics from the legacy orchestrator."""
    orchestrator = get_orchestrator()
    return orchestrator.get_execution_stats()


# ── CEO-specific endpoints (used by the dashboard) ───────────────────────────


@router.get("/ceo/decisions")
async def get_ceo_decisions(limit: int = Query(default=20, ge=1, le=200)):
    """Return recent CEO decisions for the dashboard."""
    ceo = get_ceo()
    decisions = ceo._store.all(limit)
    return {
        "decisions": [d.model_dump(mode="json") for d in decisions],
        "total": ceo._store.count(),
    }


@router.get("/ceo/status")
async def get_ceo_status():
    """Return aggregate CEO statistics for the dashboard."""
    ceo = get_ceo()
    decisions = ceo._store.all(500)
    total = len(decisions)

    mode_counts: dict = {}
    total_duration = 0.0
    for d in decisions:
        mode_counts[d.mode.value] = mode_counts.get(d.mode.value, 0) + 1
        total_duration += d.duration_ms

    avg_duration_ms = total_duration / total if total else 0.0
    mode_pct = (
        {k: round(v / total * 100, 1) for k, v in mode_counts.items()}
        if total else {}
    )

    summary = ceo.get_intuition_summary()
    success_rate = ceo._calc_success_rate()

    return {
        "total_decisions": ceo._store.count(),
        "mode_distribution": mode_pct,
        "mode_counts": mode_counts,
        "average_duration_ms": round(avg_duration_ms, 1),
        "success_rate": round(success_rate * 100, 1) if success_rate is not None else None,
        "patterns_discovered": summary.get("total_patterns", 0),
        "experts_available": 6,
        "high_confidence_patterns": summary.get("high_confidence", []),
    }


@router.post("/ceo/outcome/{decision_id}")
async def record_ceo_outcome(decision_id: str, success: bool, notes: str = ""):
    """Record the outcome of a CEO decision (for learning)."""
    ceo = get_ceo()
    updated = await ceo.record_outcome(decision_id, success=success, notes=notes)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id!r} not found")
    return updated.model_dump(mode="json")
