"""Anthropic API client (claude-3.5-sonnet, claude-3-opus, claude-3-haiku, ...)."""

import os
import json
from typing import Any, AsyncIterator, Dict, List, Union

import httpx
import structlog

from app.integrations.cloud.base import BaseCloudClient

logger = structlog.get_logger(__name__)

_MODELS = [
    {"name": "claude-3-5-sonnet-20241022", "provider": "anthropic", "tier": "powerful"},
    {"name": "claude-3-opus-20240229", "provider": "anthropic", "tier": "powerful"},
    {"name": "claude-3-haiku-20240307", "provider": "anthropic", "tier": "balanced"},
]

# Friendly aliases that callers may use → canonical model ID
_ALIASES: Dict[str, str] = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(BaseCloudClient):
    """Anthropic cloud model client using httpx."""

    provider_name = "anthropic"

    def __init__(self) -> None:
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._available = self.api_key is not None

    @property
    def is_available(self) -> bool:
        return self._available

    def _resolve_model(self, model: str) -> str:
        return _ALIASES.get(model, model)

    def _split_messages(
        self, messages: List[Dict[str, str]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Separate system prompt from user/assistant turns."""
        system = ""
        turns = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system = content
            else:
                turns.append({"role": role, "content": content})
        return system, turns

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
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        model = self._resolve_model(model)
        system, turns = self._split_messages(messages)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                return self._stream_response(client, headers, payload, model)

            response = await client.post(_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["content"][0]["text"]
            logger.info(
                "Anthropic chat completed",
                model=model,
                input_tokens=data.get("usage", {}).get("input_tokens"),
                output_tokens=data.get("usage", {}).get("output_tokens"),
            )
            return content

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncIterator[str]:
        async with client.stream("POST", _API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_str = line[6:]
                    try:
                        chunk = json.loads(chunk_str)
                        if chunk.get("type") == "content_block_delta":
                            delta = chunk.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def list_models(self) -> List[Dict[str, Any]]:
        if not self._available:
            return []
        return _MODELS

    async def health_check(self) -> Dict[str, Any]:
        if not self._available:
            return {"status": "unconfigured", "provider": self.provider_name}
        # Anthropic has no cheap ping endpoint; send a minimal message
        try:
            content = await self.chat(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return {"status": "ok", "provider": self.provider_name}
        except Exception as exc:
            return {"status": "error", "provider": self.provider_name, "error": str(exc)}
