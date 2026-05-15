"""Google Gemini API client (gemini-2.0-flash, gemini-1.5-pro, ...)."""

import os
from typing import Any, AsyncIterator, Dict, List, Union

import httpx
import structlog

from app.integrations.cloud.base import BaseCloudClient

logger = structlog.get_logger(__name__)

_MODELS = [
    {"name": "gemini-2.0-flash", "provider": "google", "tier": "balanced"},
    {"name": "gemini-1.5-pro", "provider": "google", "tier": "powerful"},
    {"name": "gemini-1.5-flash", "provider": "google", "tier": "balanced"},
]

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleClient(BaseCloudClient):
    """Google Gemini cloud model client using httpx."""

    provider_name = "google"

    def __init__(self) -> None:
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self._available = self.api_key is not None

    @property
    def is_available(self) -> bool:
        return self._available

    def _build_contents(
        self, messages: List[Dict[str, str]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Convert OpenAI-style messages to Gemini contents format."""
        system_instruction = ""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text
            else:
                # Gemini uses "user" / "model" (not "assistant")
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})
        return system_instruction, contents

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
            raise RuntimeError("GOOGLE_API_KEY is not set")

        system_instruction, contents = self._build_contents(messages)

        url = f"{_BASE_URL}/{model}:generateContent"
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        params = {"key": self.api_key}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Google API response: {data}") from exc

        logger.info("Google Gemini chat completed", model=model)
        return content

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
                    f"{_BASE_URL}",
                    params={"key": self.api_key},
                )
                response.raise_for_status()
            return {"status": "ok", "provider": self.provider_name}
        except Exception as exc:
            return {"status": "error", "provider": self.provider_name, "error": str(exc)}
