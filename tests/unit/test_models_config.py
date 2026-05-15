"""Tests for app.core.config.models_config — ModelConfig and helpers."""
import pytest
from app.core.config.models_config import (
    ModelConfig,
    DEFAULT_MODELS,
    get_model_config,
    list_model_configs,
)


def test_default_models_contains_expected_keys():
    assert "qwen2.5-coder:7b" in DEFAULT_MODELS
    assert "qwen2.5-coder:14b" in DEFAULT_MODELS
    assert "llama3.1:8b" in DEFAULT_MODELS
    assert "deepseek-coder:6.7b" in DEFAULT_MODELS


def test_model_config_default_provider_is_ollama():
    cfg = ModelConfig(name="some-model")
    assert cfg.provider == "ollama"


def test_model_config_default_cost_tier_is_local():
    cfg = ModelConfig(name="test")
    assert cfg.cost_tier == "local"


def test_get_model_config_returns_config_for_known_model():
    cfg = get_model_config("qwen2.5-coder:7b")
    assert cfg is not None
    assert cfg.name == "qwen2.5-coder:7b"
    assert "code_generation" in cfg.capabilities


def test_get_model_config_returns_none_for_unknown_model():
    assert get_model_config("unknown-model:99b") is None


def test_list_model_configs_returns_all_defaults():
    configs = list_model_configs()
    assert len(configs) == len(DEFAULT_MODELS)
    names = {c.name for c in configs}
    assert "qwen2.5-coder:7b" in names


def test_qwen_14b_has_architecture_capability():
    cfg = get_model_config("qwen2.5-coder:14b")
    assert "architecture" in cfg.capabilities


def test_llama_has_large_context_window():
    cfg = get_model_config("llama3.1:8b")
    assert cfg.context_window >= 128000


def test_model_config_capabilities_empty_by_default():
    cfg = ModelConfig(name="bare")
    assert cfg.capabilities == []
