"""Model routing — maps each AI component to its optimal local model.

Uses direct httpx calls so format="json" and thinking-token stripping work
without depending on OllamaClient's parameter signature.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Coroutine, Optional

import httpx

from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

LLMCallable = Callable[[str, str], Coroutine[Any, Any, str]]

# ── Routing table ─────────────────────────────────────────────────────────────

MODEL_ROUTING: dict[str, str] = {
    "mode_selection":        "phi4-mini:latest",      # fast classification
    "council_proposal":      "dolphin-mistral:7b",    # general instruction following
    "council_critique":      "dolphin-mistral:7b",
    "council_vote":          "dolphin-mistral:7b",
    "synthesis":             "deepseek-r1:7b",        # actual reasoning over conflicts
    "code_generation":       "dolphincoder:7b",       # fine-tuned for code
    "pattern_discovery":     "qwen3.5:9b",            # analytical reflection
    "architecture_analysis": "gemma4:e4b",            # largest — complex structural analysis
    "security_audit":        "dolphin-mistral:7b",
    "performance_analysis":  "dolphin-mistral:7b",
    "ceo_reasoning":         "dolphin-mistral:7b",
    "default":               "dolphin-mistral:7b",
}

# Models that output <think>...</think> before their real answer
_THINKING_MODELS = {"deepseek-r1"}

# Models that CANNOT receive format="json" (vision, embedding, or thinking models)
_NO_JSON_FORMAT = {"deepseek-r1", "bge-m3", "qwen2.5vl"}


def _needs_think_strip(model: str) -> bool:
    return any(m in model.lower() for m in _THINKING_MODELS)


def _supports_json_format(model: str) -> bool:
    return not any(m in model.lower() for m in _NO_JSON_FORMAT)


def strip_thinking_tokens(text: str) -> str:
    """Remove <think>…</think> blocks emitted by deepseek-r1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── LLM factory ───────────────────────────────────────────────────────────────

def make_ollama_llm(model: str) -> LLMCallable:
    """Return a model-specific async LLM callable.

    - Applies format="json" for compatible models (deterministic output).
    - Strips thinking tokens for deepseek-r1.
    - Temperature 0.1 for reproducible structured responses.
    """

    async def _llm(system: str, prompt: str) -> str:
        from app.core.config.settings import settings

        base_url = getattr(settings, "ollama", None)
        base_url = (
            base_url.base_url if base_url and hasattr(base_url, "base_url")
            else "http://localhost:11434"
        ).rstrip("/")

        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1},
            "stream": False,
        }

        if _supports_json_format(model):
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                raw = data.get("message", {}).get("content", "") or str(data)
        except Exception as exc:
            logger.warning(
                "Direct Ollama call failed, falling back to OllamaClient",
                model=model,
                error=str(exc),
            )
            # Fallback to OllamaClient (without format param)
            from app.integrations.ollama.client import get_default_client
            raw = await get_default_client().chat(
                model=model,
                messages=payload["messages"],
                temperature=0.1,
            )
            raw = raw if isinstance(raw, str) else str(raw)

        if _needs_think_strip(model):
            raw = strip_thinking_tokens(raw)

        return raw

    return _llm


# ── JSON retry wrapper ────────────────────────────────────────────────────────

async def call_with_json_retry(
    llm: LLMCallable,
    system: str,
    user: str,
    parse_fn,
    max_retries: int = 2,
):
    """Call LLM, parse result with parse_fn; retry once with corrective prompt.

    Returns the parsed result or None if all retries fail.
    parse_fn must return None (not raise) on parse failure.
    """
    last_raw = ""
    current_user = user

    for attempt in range(max_retries):
        try:
            last_raw = await llm(system, current_user)
            result = parse_fn(last_raw)
            if result is not None:
                return result
        except Exception as exc:
            logger.debug("LLM call failed in retry loop", attempt=attempt, error=str(exc))

        if attempt == 0:
            snippet = last_raw[:300] if last_raw else "(no response received)"
            current_user = (
                "Your previous response could not be parsed as valid JSON.\n\n"
                f"Previous response (first 300 chars):\n{snippet}\n\n"
                "Return ONLY a valid JSON object. No markdown fences. No explanation. No prose."
            )
            logger.debug("Retrying with corrective prompt")

    logger.warning("All JSON retries exhausted — caller will use fallback")
    return None
