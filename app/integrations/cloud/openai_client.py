"""OpenAI API client (gpt-4o, o1, gpt-4o-mini, ...)."""

import os
import json
from typing import Any, AsyncIterator, Dict, List, Union

import httpx
import structlog

from app.integrations.cloud.base import BaseCloudClient

logger = structlog.get_logger(__name__)

_MODELS = [
    {"name": "gpt-4o", "provider": "openai", "tier": "powerful"},
    {"name": "gpt-4o-mini", "provider": "openai", "tier": "balanced"},
    {"name": "o1", "provider": "openai", "tier": "powerful"},
    {"name": "o1-mini", "provider": "openai", "tier": "balanced"},
]


class OpenAIClient(BaseCloudClient):
    """OpenAI cloud model client using httpx."""

    provider_name = "openai"

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self._available = self.api_key is not None

    @property
    def is_available(self) -> bool:
        return self._available

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
            raise RuntimeError("OPENAI_API_KEY is not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # o1 models do not support temperature
        if not model.startswith("o1"):
            payload["temperature"] = temperature

        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                return self._stream_response(client, headers, payload, model)

            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info(
                "OpenAI chat completed",
                model=model,
                tokens=data.get("usage", {}).get("total_tokens"),
            )
            return content

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncIterator[str]:
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
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
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
            return {"status": "ok", "provider": self.provider_name}
        except Exception as exc:
            return {"status": "error", "provider": self.provider_name, "error": str(exc)}
