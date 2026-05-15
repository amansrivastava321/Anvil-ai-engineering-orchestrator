"""
Model management endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.model_service import (
    get_model_service,
    TaskCategory,
    ModelTier,
)

router = APIRouter()


@router.get("/")
async def list_models(
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    tier: Optional[str] = Query(None, description="Filter by capability tier"),
):
    """
    List available models, optionally filtered by task or tier.

    Returns models grouped into ``local`` (Ollama) and ``cloud`` (API-hosted)
    lists, plus a flat ``models`` list for backward compatibility.

    Task types: code_generation, code_review, debugging, architecture_analysis, etc.
    Tiers: fast, balanced, powerful, specialized
    """
    service = get_model_service()

    task = TaskCategory(task_type) if task_type else None
    model_tier = ModelTier(tier) if tier else None

    # Flat list (backward-compatible)
    models = await service.list_available_models(
        task_type=task,
        tier=model_tier,
    )

    # Grouped view (new)
    grouped = await service.list_all_models_grouped()

    return {
        "models": models,
        "local": grouped["local"],
        "cloud": grouped["cloud"],
        "total": len(models),
    }


@router.get("/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    service = get_model_service()
    info = await service.get_model_info(model_name)
    
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return info


@router.get("/health")
async def models_health():
    """Health check for all registered models."""
    service = get_model_service()
    return await service.health_check()


@router.get("/stats/selection")
async def selection_stats():
    """Get model selection statistics."""
    service = get_model_service()
    return service.get_selection_stats()