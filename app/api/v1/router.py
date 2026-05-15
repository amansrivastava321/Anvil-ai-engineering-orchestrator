"""
API v1 router - aggregates all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.agent import router as agent_router
from app.api.v1.endpoints.models import router as models_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.monitor import router as monitor_router
from app.api.v1.endpoints.evolution import router as evolution_router
from app.api.v1.endpoints.repo import router as repo_router

api_router = APIRouter()

api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(models_router, prefix="/models", tags=["models"])
api_router.include_router(health_router, prefix="/system", tags=["system"])
api_router.include_router(monitor_router, prefix="/monitor", tags=["monitor"])
api_router.include_router(evolution_router, prefix="/evolution", tags=["evolution"])
api_router.include_router(repo_router, prefix="/repo", tags=["repo"])