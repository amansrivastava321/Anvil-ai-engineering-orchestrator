"""Repository onboarding and scanning endpoints."""

from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.onboarding_service import get_onboarding_service
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class AnalyzeRequest(BaseModel):
    repo_path: str
    force_full: bool = False   # bypass state check and re-run everything


class OnboardRequest(BaseModel):
    repo_path: str
    force_rebuild: bool = False


class QuickScanRequest(BaseModel):
    repo_path: str


# ── Smart entry point ─────────────────────────────────────────────────────────


@router.post("/analyze")
async def analyze_repository(request: AnalyzeRequest):
    """Smart analysis that respects repo memory.

    - First time: runs full Graphify + CEO analysis, saves state.
    - Subsequent calls: returns current state + detected changes immediately.
    - Use force_full=true to re-run the full pipeline regardless.

    Returns status="new_analysis" | "previously_analyzed".
    Previously-analyzed repos return immediately with options for next steps.
    """
    svc = get_onboarding_service()
    try:
        result = await svc.analyze_repository(
            repo_path=request.repo_path,
            force_full=request.force_full,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Analysis failed", repo=request.repo_path, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Full onboarding (force) ───────────────────────────────────────────────────


@router.post("/onboard")
async def onboard_repository(request: OnboardRequest, background_tasks: BackgroundTasks):
    """Force a full onboarding pipeline regardless of prior state.

    Graphify → graph query → CEO analysis → save state → return report.
    May take 30–120 seconds for large repositories.
    """
    svc = get_onboarding_service()
    try:
        report = await svc.onboard_repository(
            repo_path=request.repo_path,
            force_rebuild=request.force_rebuild,
        )
        return report.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Onboarding failed", repo=request.repo_path, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Quick scan ────────────────────────────────────────────────────────────────


@router.post("/quick-scan")
async def quick_scan(request: QuickScanRequest):
    """Incremental scan: only analyses files changed since last analysis.

    Uses git diff against the last analyzed commit hash. Much faster than a
    full onboarding. Returns a delta report (only what changed).
    """
    svc = get_onboarding_service()
    try:
        report = await svc.quick_scan(repo_path=request.repo_path)
        return report.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Quick scan failed", repo=request.repo_path, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── State and history ─────────────────────────────────────────────────────────


@router.get("/state/{repo_path:path}")
async def get_repo_state(repo_path: str):
    """Return the full persistent state for a repository (all history, trends, etc.)."""
    svc = get_onboarding_service()
    state = svc.get_state(repo_path)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {repo_path!r} not yet analyzed. Call /analyze first.",
        )
    # Include computed properties (pydantic @property values don't serialize automatically)
    data = state.model_dump(mode="json")
    data["trend"] = state.trend
    data["trend_arrow"] = state.trend_arrow
    data["improvement_rate"] = state.improvement_rate
    data["needs_attention"] = state.needs_attention
    data["total_issues"] = state.total_issues()
    data["days_since_analysis"] = state.days_since_analysis()
    return data


@router.get("/list")
async def list_repos():
    """Return all known repositories as rich summary cards with health trends."""
    svc = get_onboarding_service()
    repos = svc.list_repos()
    summary = svc.get_weekly_summary()
    return {
        "repos": repos,
        "total": len(repos),
        "weekly_summary": summary,
    }


@router.get("/report/{repo_path:path}")
async def get_repo_report(repo_path: str):
    """Return the last full onboarding report for a repository (in-memory only)."""
    svc = get_onboarding_service()
    report = svc.get_report(repo_path)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report in memory for {repo_path!r}. Run /analyze or /onboard first.",
        )
    return report.model_dump(mode="json")


@router.get("/summary")
async def weekly_summary():
    """Return this week's aggregate stats: repos monitored, commits, issues fixed."""
    svc = get_onboarding_service()
    return svc.get_weekly_summary()
