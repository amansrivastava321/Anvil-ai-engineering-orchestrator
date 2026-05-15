"""Embedding helpers — bge-m3 via Ollama /api/embeddings."""
from __future__ import annotations

from typing import List

import httpx
import numpy as np

from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "bge-m3:latest"
_TIMEOUT = 30.0


async def get_embedding(text: str, model: str = _DEFAULT_MODEL) -> List[float]:
    """Return an embedding vector for text using bge-m3 via Ollama."""
    from app.core.config.settings import settings

    base_url = getattr(settings, "ollama", None)
    base_url = (
        base_url.base_url if base_url and hasattr(base_url, "base_url")
        else "http://localhost:11434"
    ).rstrip("/")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
