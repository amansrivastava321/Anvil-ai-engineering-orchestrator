"""Monitor API endpoints — start/stop proactive repository monitoring."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.monitoring.logging import get_logger
from app.services.proactive_service import get_proactive_service

logger = get_logger(__name__)
router = APIRouter()


# ── Request / response schemas ─────────────────────────────────────────────────

class MonitorStartRequest(BaseModel):
    repo_paths: List[str] = Field(
        ...,
        min_length=1,
        description="Absolute paths to git repositories to watch",
    )
    poll_interval: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Poll interval in seconds",
    )
    auto_debug: bool = Field(
        default=True,
        description="Automatically trigger AI diagnosis on test failures",
    )


class MonitorStopRequest(BaseModel):
    repo_paths: Optional[List[str]] = Field(
        default=None,
        description="Repositories to stop watching. None means stop all.",
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/start", summary="Start monitoring repositories")
async def start_monitoring(request: MonitorStartRequest) -> Dict[str, Any]:
    """Begin proactive monitoring for the given repository paths.

    Already-watched repositories are skipped — calling this endpoint is
    idempotent.
    """
    service = get_proactive_service()
    try:
        result = await service.start_watching(
            repos=request.repo_paths,
            poll_interval=request.poll_interval,
            auto_debug=request.auto_debug,
        )
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("Failed to start monitoring", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stop", summary="Stop monitoring repositories")
async def stop_monitoring(request: MonitorStopRequest) -> Dict[str, Any]:
    """Stop monitoring specific repositories, or all if ``repo_paths`` is
    omitted.
    """
    service = get_proactive_service()
    try:
        result = await service.stop_watching(repos=request.repo_paths)
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("Failed to stop monitoring", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status", summary="Get current monitoring status")
async def monitor_status() -> Dict[str, Any]:
    """Return real-time status for all watched repositories."""
    service = get_proactive_service()
    try:
        return await service.get_status()
    except Exception as exc:
        logger.error("Failed to retrieve monitor status", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
