"""Model configuration — defines available models and their capabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    name: str
    provider: str = "ollama"
    context_window: int = 8192
    supports_tools: bool = False
    capabilities: List[str] = field(default_factory=list)
    cost_tier: str = "local"  # local, api_cheap, api_expensive


# Default model configurations
DEFAULT_MODELS: Dict[str, ModelConfig] = {
    "qwen2.5-coder:7b": ModelConfig(
        name="qwen2.5-coder:7b",
        capabilities=["code_generation", "code_review", "debugging", "testing"],
        context_window=32768,
    ),
    "qwen2.5-coder:14b": ModelConfig(
        name="qwen2.5-coder:14b",
        capabilities=["code_generation", "code_review", "debugging", "architecture"],
        context_window=32768,
    ),
    "llama3.1:8b": ModelConfig(
        name="llama3.1:8b",
        capabilities=["general_qa", "documentation", "report"],
        context_window=128000,
    ),
    "deepseek-coder:6.7b": ModelConfig(
        name="deepseek-coder:6.7b",
        capabilities=["code_generation", "debugging"],
        context_window=16384,
    ),
}


def get_model_config(name: str) -> Optional[ModelConfig]:
    return DEFAULT_MODELS.get(name)


def list_model_configs() -> List[ModelConfig]:
    return list(DEFAULT_MODELS.values())
