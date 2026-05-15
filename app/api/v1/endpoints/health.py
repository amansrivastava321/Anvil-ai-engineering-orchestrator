"""
System health and status endpoints.
"""

from datetime import datetime
from fastapi import APIRouter

from app.core.config.settings import settings
from app.services.orchestrator import get_orchestrator

router = APIRouter()


@router.get("/status")
async def system_status():
    """Get comprehensive system status including Ollama, CEO, and orchestrator health."""
    orchestrator = get_orchestrator()
    base = await orchestrator.health_check()

    # Attach CEO status so the dashboard gets everything in one call
    try:
        from app.ai.ceo import get_ceo
        ceo = get_ceo()
        summary = ceo.get_intuition_summary()
        base["ceo"] = {
            "status": "online",
            "total_decisions": summary.get("total_decisions", 0),
            "patterns_discovered": summary.get("total_patterns", 0),
            "success_rate": summary.get("success_rate"),
        }
    except Exception as e:
        base["ceo"] = {"status": "unavailable", "error": str(e)}

    return base


@router.get("/stats")
async def system_stats():
    """Get execution statistics."""
    orchestrator = get_orchestrator()
    return orchestrator.get_execution_stats()


@router.get("/info")
async def system_info():
    """Get basic system information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment.value,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    Returns 200 if the system is ready to accept requests.
    """
    try:
        orchestrator = get_orchestrator()
        health = await orchestrator.ollama.health_check()
        
        if health.get("status") == "healthy":
            return {"status": "ready"}
        return {"status": "not_ready", "reason": "Ollama not healthy"}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    Returns 200 if the server is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}