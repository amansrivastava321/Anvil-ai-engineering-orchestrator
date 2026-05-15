"""OpenRouter client — single API key gives access to 200+ models."""

import os
import json
from typing import Any, AsyncIterator, Dict, List, Union

import httpx
import structlog

from app.integrations.cloud.base import BaseCloudClient

logger = structlog.get_logger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"

_MODELS = [
    {"name": "openai/gpt-4o", "provider": "openrouter", "tier": "powerful"},
    {"name": "anthropic/claude-3.5-sonnet", "provider": "openrouter", "tier": "powerful"},
    {"name": "google/gemini-2.0-flash", "provider": "openrouter", "tier": "balanced"},
    {"name": "deepseek/deepseek-r1", "provider": "openrouter", "tier": "powerful"},
    {"name": "meta-llama/llama-3.3-70b-instruct", "provider": "openrouter", "tier": "powerful"},
]


class OpenRouterClient(BaseCloudClient):
    """
    OpenRouter cloud model client.

    OpenRouter is OpenAI API-compatible; it routes to 200+ underlying models
    using model names in the format ``provider/model-name``.
    One API key is all that's needed.
    """

    provider_name = "openrouter"

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self._available = self.api_key is not None

    @property
    def is_available(self) -> bool:
        return self._available

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/anvil-ai",
            "X-Title": "Anvil",
        }

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, AsyncIterator[str]]:
        if not self._available:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                return self._stream_response(client, payload, model)

            response = await client.post(
                f"{_BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info(
                "OpenRouter chat completed",
                model=model,
                tokens=data.get("usage", {}).get("total_tokens"),
            )
            return content

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncIterator[str]:
        async with client.stream(
            "POST",
            f"{_BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_str = line[6:]
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def list_models(self) -> List[Dict[str, Any]]:
        if not self._available:
            return []
        return _MODELS

    async def health_check(self) -> Dict[str, Any]:
        if not self._available:
            return {"status": "unconfigured", "provider": self.provider_name}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{_BASE_URL}/models",
                    headers=self._headers(),
                )
                response.raise_for_status()
            return {"status": "ok", "provider": self.provider_name}
        except Exception as exc:
            return {"status": "error", "provider": self.provider_name, "error": str(exc)}
