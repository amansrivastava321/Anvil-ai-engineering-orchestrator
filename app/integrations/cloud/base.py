"""Base interface for all cloud model provider clients."""

from typing import Any, AsyncIterator, Dict, List, Union


class BaseCloudClient:
    """
    Base interface for all cloud model providers.

    Each provider subclass implements chat(), list_models(), and health_check().
    The routing layer in OllamaClient calls these methods transparently so
    callers never need to know whether a model is local or cloud-hosted.
    """

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, AsyncIterator[str]]:
        """Send a chat request. All providers implement this."""
        raise NotImplementedError

    async def list_models(self) -> List[Dict[str, Any]]:
        """Return available models for this provider."""
        raise NotImplementedError

    async def health_check(self) -> Dict[str, Any]:
        """Check if the provider API is reachable and authenticated."""
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        """Return provider identifier: openai, anthropic, google, openrouter, etc."""
        raise NotImplementedError

    @property
    def is_available(self) -> bool:
        """True if the provider API key is configured."""
        raise NotImplementedError
