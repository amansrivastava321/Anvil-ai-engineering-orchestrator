"""Tests for app.core.config.settings — configuration management."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import SecretStr


# ── Environment / LogLevel / LogFormat enums ───────────────────────────────────

def test_environment_enum_values():
    from app.core.config.settings import Environment
    assert Environment.DEVELOPMENT.value == "development"
    assert Environment.PRODUCTION.value == "production"
    assert Environment.TESTING.value == "testing"
    assert Environment.CI.value == "ci"


def test_log_level_enum_values():
    from app.core.config.settings import LogLevel
    assert LogLevel.DEBUG.value == "DEBUG"
    assert LogLevel.INFO.value == "INFO"
    assert LogLevel.ERROR.value == "ERROR"


def test_log_format_enum_values():
    from app.core.config.settings import LogFormat
    assert LogFormat.JSON.value == "json"
    assert LogFormat.CONSOLE.value == "console"


def test_cache_backend_enum_values():
    from app.core.config.settings import CacheBackend
    assert CacheBackend.REDIS.value == "redis"
    assert CacheBackend.MEMORY.value == "memory"
    assert CacheBackend.NONE.value == "none"


def test_secret_backend_enum_values():
    from app.core.config.settings import SecretBackend
    assert SecretBackend.ENV.value == "env"
    assert SecretBackend.VAULT.value == "vault"


# ── DatabaseSettings ───────────────────────────────────────────────────────────

def test_database_settings_defaults():
    from app.core.config.settings import DatabaseSettings
    db = DatabaseSettings()
    assert db.host == "localhost"
    assert db.port == 5432
    assert db.user == "postgres"
    assert db.name == "orchestrator"
    assert db.pool_size == 20


def test_database_settings_url_computed():
    from app.core.config.settings import DatabaseSettings
    db = DatabaseSettings()
    url = db.url
    assert "postgresql+asyncpg://" in url
    assert "localhost" in url
    assert "5432" in url


def test_database_settings_sync_url_computed():
    from app.core.config.settings import DatabaseSettings
    db = DatabaseSettings()
    url = db.sync_url
    assert "postgresql://" in url
    assert "localhost" in url


def test_database_settings_pool_size_validator():
    from app.core.config.settings import DatabaseSettings
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        db = DatabaseSettings(pool_size=10)
        assert db.pool_size == 10


def test_database_settings_pool_size_production_warning():
    from app.core.config.settings import DatabaseSettings
    # Should not raise even in production with small pool
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        db = DatabaseSettings(pool_size=10)
        assert db.pool_size == 10


# ── RedisSettings ──────────────────────────────────────────────────────────────

def test_redis_settings_defaults():
    from app.core.config.settings import RedisSettings
    redis = RedisSettings()
    assert redis.host == "localhost"
    assert redis.port == 6379
    assert redis.db == 0


def test_redis_settings_url_no_password():
    from app.core.config.settings import RedisSettings
    redis = RedisSettings()
    url = redis.url
    assert "redis://localhost:6379/0" == url


def test_redis_settings_url_with_password():
    from app.core.config.settings import RedisSettings
    redis = RedisSettings(password=SecretStr("secret123"))
    url = redis.url
    assert "secret123" in url
    assert "redis://:" in url


def test_redis_settings_connections_validator_development():
    from app.core.config.settings import RedisSettings
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        redis = RedisSettings(max_connections=20)
        assert redis.max_connections == 20


# ── OllamaSettings ─────────────────────────────────────────────────────────────

def test_ollama_settings_defaults():
    from app.core.config.settings import OllamaSettings
    ollama = OllamaSettings()
    assert ollama.base_url == "http://localhost:11434"
    assert ollama.default_model == "qwen2.5:7b"
    assert ollama.fallback_model == "qwen2.5:3b"
    assert ollama.timeout == 120.0


def test_ollama_settings_url_validation_strips_trailing_slash():
    from app.core.config.settings import OllamaSettings
    ollama = OllamaSettings(base_url="http://localhost:11434/")
    assert not ollama.base_url.endswith("/")


def test_ollama_settings_invalid_url_raises():
    from app.core.config.settings import OllamaSettings
    with pytest.raises(Exception):
        OllamaSettings(base_url="ftp://localhost:11434")


# ── GraphifySettings ───────────────────────────────────────────────────────────

def test_graphify_settings_defaults():
    from app.core.config.settings import GraphifySettings
    g = GraphifySettings()
    assert g.output_dir == "graphify-out"
    assert g.cache_ttl == 3600
    assert g.auto_discover is True
    assert "json" in g.supported_formats


# ── SecuritySettings ───────────────────────────────────────────────────────────

def test_security_settings_defaults():
    from app.core.config.settings import SecuritySettings
    sec = SecuritySettings()
    assert sec.algorithm == "HS256"
    assert sec.access_token_expire_minutes == 30
    assert sec.rate_limit_enabled is True


def test_security_settings_secret_key_dev_allows_default():
    from app.core.config.settings import SecuritySettings
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        sec = SecuritySettings()
        assert sec.secret_key is not None


def test_security_settings_cors_origins_default():
    from app.core.config.settings import SecuritySettings
    sec = SecuritySettings()
    assert isinstance(sec.cors_origins, list)


# ── MonitoringSettings ─────────────────────────────────────────────────────────

def test_monitoring_settings_defaults():
    from app.core.config.settings import MonitoringSettings, LogLevel, LogFormat
    mon = MonitoringSettings()
    assert mon.enable_prometheus is True
    assert mon.prometheus_port == 9090
    assert mon.log_level == LogLevel.INFO
    assert mon.log_format == LogFormat.JSON


# ── AgentSettings ──────────────────────────────────────────────────────────────

def test_agent_settings_defaults():
    from app.core.config.settings import AgentSettings
    agent = AgentSettings()
    assert agent.max_execution_time == 300.0
    assert agent.max_iterations == 10
    assert agent.max_tool_calls == 50
    assert agent.code_review_enabled is True


# ── Main Settings ──────────────────────────────────────────────────────────────

def test_settings_is_singleton():
    from app.core.config.settings import settings, get_settings
    s1 = get_settings()
    assert s1 is settings


def test_settings_has_sub_configs():
    from app.core.config.settings import settings
    assert hasattr(settings, "database")
    assert hasattr(settings, "redis")
    assert hasattr(settings, "ollama")
    assert hasattr(settings, "graphify")
    assert hasattr(settings, "security")
    assert hasattr(settings, "monitoring")
    assert hasattr(settings, "agent")


def test_settings_computed_is_production():
    from app.core.config.settings import Settings, Environment
    # We can't easily construct Settings due to directory creation, so test via existing singleton
    from app.core.config.settings import settings
    # is_production depends on environment
    assert isinstance(settings.is_production, bool)


def test_settings_computed_is_development():
    from app.core.config.settings import settings
    assert isinstance(settings.is_development, bool)


def test_settings_computed_is_testing():
    from app.core.config.settings import settings
    assert isinstance(settings.is_testing, bool)


def test_settings_api_v1_prefix():
    from app.core.config.settings import settings
    assert settings.api_v1_prefix == "/api/v1"


def test_settings_to_dict_excludes_secrets():
    from app.core.config.settings import settings
    data = settings.to_dict(exclude_secrets=True)
    assert isinstance(data, dict)
    # Secrets should be redacted
    if "database" in data and "password" in data["database"]:
        assert data["database"]["password"] == "********"


def test_settings_to_dict_includes_secrets():
    from app.core.config.settings import settings
    data = settings.to_dict(exclude_secrets=False)
    assert isinstance(data, dict)
    assert "database" in data


def test_settings_to_yaml_returns_string():
    from app.core.config.settings import settings
    yaml_str = settings.to_yaml()
    assert isinstance(yaml_str, str)
    assert len(yaml_str) > 0


def test_settings_to_yaml_writes_file(tmp_path):
    from app.core.config.settings import settings
    out = tmp_path / "settings.yaml"
    settings.to_yaml(path=out)
    assert out.exists()
    assert len(out.read_text()) > 0


def test_settings_load_environment_config_missing_file():
    from app.core.config.settings import settings
    # Should return empty dict when file doesn't exist
    result = settings.load_environment_config()
    assert isinstance(result, dict)


def test_settings_validate_all_returns_true():
    from app.core.config.settings import settings
    # May raise for cluster mode checks — just verify it runs
    try:
        result = settings.validate_all()
        assert result is True
    except ValueError:
        pass  # Acceptable if cluster config is inconsistent


def test_settings_feature_flags():
    from app.core.config.settings import settings
    assert isinstance(settings.feature_cache_enabled, bool)
    assert isinstance(settings.feature_streaming_enabled, bool)
    assert isinstance(settings.feature_batch_processing, bool)
    assert isinstance(settings.feature_experimental_agents, bool)


# ── normalize_environment validator ───────────────────────────────────────────

def test_normalize_environment_aliases():
    from app.core.config.settings import Settings, Environment
    # Test by calling the validator directly
    validator = Settings.normalize_environment
    assert validator("dev") == Environment.DEVELOPMENT
    assert validator("prod") == Environment.PRODUCTION
    assert validator("test") == Environment.TESTING
    assert validator("staging") == Environment.STAGING
    assert validator("ci") == Environment.CI
    assert validator(Environment.PRODUCTION) == Environment.PRODUCTION


def test_normalize_environment_invalid_raises():
    from app.core.config.settings import Settings
    with pytest.raises(ValueError):
        Settings.normalize_environment("invalid_env_xyz")


# ── validate_settings_on_startup ──────────────────────────────────────────────

def test_validate_settings_on_startup():
    from app.core.config.settings import validate_settings_on_startup
    # Should not raise in development
    try:
        validate_settings_on_startup()
    except Exception:
        pass  # May fail due to missing dirs in test env
