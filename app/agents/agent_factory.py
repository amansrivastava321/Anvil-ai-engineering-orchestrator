"""Agent factory — creates and caches specialized agents by type name.

Registry uses lazy imports to avoid circular imports and allow the factory
to be instantiated before all agent modules are loaded.
"""

import importlib
from typing import Type

import structlog

from app.agents.base_agent import BaseAgent
from app.integrations.ollama.client import OllamaClient, get_default_client

logger = structlog.get_logger(__name__)

# Maps agent type name → dotted import path "module:ClassName"
_REGISTRY: dict[str, str] = {
    "code": "app.agents.specialized.code_agent:CodeAgent",
    "architecture": "app.agents.specialized.architecture_agent:ArchitectureAgent",
    "testing": "app.agents.specialized.testing_agent:TestingAgent",
    "documentation": "app.agents.specialized.documentation_agent:DocumentationAgent",
    "security": "app.agents.specialized.security_agent:SecurityAgent",
    "performance": "app.agents.specialized.performance_agent:PerformanceAgent",
}


class AgentFactory:
    """Creates and caches agent instances by type name."""

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        self._ollama = ollama_client or get_default_client()
        self._cache: dict[str, BaseAgent] = {}

    def get_agent(self, agent_type: str) -> BaseAgent:
        """Return a cached agent instance for the given type.

        Raises KeyError if agent_type is not registered.
        Raises ImportError if the agent module cannot be loaded.
        """
        if agent_type in self._cache:
            return self._cache[agent_type]

        if agent_type not in _REGISTRY:
            raise KeyError(
                f"Unknown agent type: {agent_type!r}. Available: {list(_REGISTRY)}"
            )

        agent_class = self._load_agent_class(agent_type)
        agent = agent_class(ollama_client=self._ollama)
        self._cache[agent_type] = agent
        logger.info("Agent created", agent_type=agent_type, agent_class=agent_class.__name__)
        return agent

    def list_agents(self) -> dict[str, str]:
        """Return registered agent types mapped to their class import paths."""
        return dict(_REGISTRY)

    def _load_agent_class(self, agent_type: str) -> Type[BaseAgent]:
        """Dynamically import the agent class for the given type."""
        path = _REGISTRY[agent_type]
        module_path, class_name = path.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)  # type: ignore[return-value]


_factory: AgentFactory | None = None


def get_agent_factory(ollama_client: OllamaClient | None = None) -> AgentFactory:
    """Return the global AgentFactory singleton."""
    global _factory
    if _factory is None:
        _factory = AgentFactory(ollama_client=ollama_client)
    return _factory
