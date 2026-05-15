"""
Auto-discovers which cloud providers are configured and makes them available.

No config files needed. Users just set an environment variable:

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=AIza...
    export OPENROUTER_API_KEY=sk-or-...

Anvil detects them at startup and adds those providers' models to the pool.
If no API keys are set, this registry is empty and everything continues
working with local Ollama models — no errors, no warnings.
"""

import importlib
from typing import Dict, List, Optional

import structlog

from app.integrations.cloud.base import BaseCloudClient
from app.integrations.cloud.openai_client import OpenAIClient
from app.integrations.cloud.anthropic_client import AnthropicClient
from app.integrations.cloud.google_client import GoogleClient
from app.integrations.cloud.openrouter_client import OpenRouterClient

logger = structlog.get_logger(__name__)


def _static_models(provider: BaseCloudClient) -> List[Dict]:
    """
    Return the in-memory _MODELS list from a provider's module.

    Each provider module defines a module-level _MODELS constant so this
    lookup is always synchronous — no I/O, no event loop needed.
    """
    mod = importlib.import_module(type(provider).__module__)
    return list(getattr(mod, "_MODELS", []))


class CloudRegistry:
    """
    Auto-detects configured cloud providers from environment variables
    and exposes their models alongside local Ollama models.
    """

    def __init__(self) -> None:
        self._providers: List[BaseCloudClient] = []
        # model_name → provider client; built once on first lookup
        self._model_index: Optional[Dict[str, BaseCloudClient]] = None
        self._discover_providers()

    def _discover_providers(self) -> None:
        """Check which API keys are set and register those providers."""
        candidates: List[BaseCloudClient] = [
            OpenAIClient(),
            AnthropicClient(),
            GoogleClient(),
            OpenRouterClient(),
        ]
        for provider in candidates:
            if provider.is_available:
                self._providers.append(provider)
                logger.info(
                    "Cloud provider configured",
                    provider=provider.provider_name,
                )

        if not self._providers:
            logger.debug("No cloud providers configured — using local models only")

    def _build_index(self) -> Dict[str, BaseCloudClient]:
        """Build model-name → client mapping from all registered providers."""
        index: Dict[str, BaseCloudClient] = {}
        for provider in self._providers:
            for model in _static_models(provider):
                index[model["name"]] = provider
        return index

    @property
    def _index(self) -> Dict[str, BaseCloudClient]:
        if self._model_index is None:
            self._model_index = self._build_index()
        return self._model_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_client_for_model(self, model_name: str) -> Optional[BaseCloudClient]:
        """Return the provider client that owns *model_name*, or None."""
        return self._index.get(model_name)

    def get_all_models(self) -> List[Dict]:
        """Return all available models from all configured providers."""
        models: List[Dict] = []
        for provider in self._providers:
            models.extend(_static_models(provider))
        return models

    def has_cloud_models(self) -> bool:
        """True if any cloud provider is configured."""
        return bool(self._providers)

    def provider_health(self) -> Dict[str, str]:
        """Return configured provider names → 'configured' status."""
        return {p.provider_name: "configured" for p in self._providers}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: Optional[CloudRegistry] = None


def get_cloud_registry() -> CloudRegistry:
    """Get or create the singleton CloudRegistry."""
    global _registry
    if _registry is None:
        _registry = CloudRegistry()
    return _registry
