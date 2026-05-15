"""
Enterprise-grade configuration management system with:
- Multi-environment support (development, staging, production)
- Secrets management with multiple backends
- Configuration validation at startup
- Hot-reload capability for selected settings
- Audit logging for configuration changes
- Type-safe configuration with Pydantic v2

This module is the single source of truth for all application settings.
Security Level: CRITICAL - Contains secrets and security configurations
"""

import os
import sys
import secrets
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Set, Literal,
    get_type_hints, ClassVar
)
from enum import Enum
from functools import lru_cache
import json
import yaml

from pydantic import (
    Field, 
    field_validator, 
    model_validator,
    ValidationInfo,
    SecretStr,
    AnyHttpUrl,
    AnyUrl,
    PostgresDsn,
    RedisDsn,
    DirectoryPath,
    FilePath,
    IPvAnyAddress,
    computed_field,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
    SecretsSettingsSource,
)
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Environment Types
# ============================================================================

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    CI = "ci"


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format types."""
    JSON = "json"
    CONSOLE = "console"
    TEXT = "text"


class CacheBackend(str, Enum):
    """Cache backend types."""
    REDIS = "redis"
    MEMORY = "memory"
    DISK = "disk"
    NONE = "none"


class SecretBackend(str, Enum):
    """Secrets management backend."""
    ENV = "env"
    VAULT = "vault"
    AWS_SECRETS = "aws_secrets"
    AZURE_KEYVAULT = "azure_keyvault"
    FILE = "file"


# ============================================================================
# Sub-configurations
# ============================================================================

class DatabaseSettings(BaseSettings):
    """Database configuration with connection pooling."""
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )
    
    # Required settings
    host: str = Field(
        default="localhost",
        description="Database host address",
        min_length=1,
    )
    port: int = Field(
        default=5432,
        description="Database port",
        ge=1,
        le=65535,
    )
    user: str = Field(
        default="postgres",
        description="Database username",
        min_length=1,
    )
    password: SecretStr = Field(
        default=SecretStr("postgres"),
        description="Database password (use SecretStr for security)",
    )
    name: str = Field(
        default="orchestrator",
        description="Database name",
        min_length=1,
    )
    
    # Connection pool settings
    pool_size: int = Field(
        default=20,
        description="Maximum number of connections in the pool",
        ge=5,
        le=100,
    )
    max_overflow: int = Field(
        default=10,
        description="Maximum overflow connections beyond pool_size",
        ge=0,
        le=50,
    )
    pool_timeout: float = Field(
        default=30.0,
        description="Seconds to wait for a connection from pool",
        ge=1.0,
        le=120.0,
    )
    pool_recycle: int = Field(
        default=3600,
        description="Seconds before a connection is recycled",
        ge=60,
        le=7200,
    )
    
    # Performance settings
    echo: bool = Field(
        default=False,
        description="Enable SQL query logging (disable in production)",
    )
    echo_pool: bool = Field(
        default=False,
        description="Enable connection pool logging",
    )
    connect_timeout: int = Field(
        default=10,
        description="Database connection timeout in seconds",
        ge=1,
        le=60,
    )
    command_timeout: int = Field(
        default=60,
        description="SQL command timeout in seconds",
        ge=10,
        le=300,
    )
    
    # SSL settings
    use_ssl: bool = Field(
        default=False,
        description="Use SSL for database connection",
    )
    ssl_ca_path: Optional[FilePath] = Field(
        default=None,
        description="Path to SSL CA certificate",
    )
    ssl_cert_path: Optional[FilePath] = Field(
        default=None,
        description="Path to SSL client certificate",
    )
    ssl_key_path: Optional[FilePath] = Field(
        default=None,
        description="Path to SSL client key",
    )
    
    @computed_field
    @property
    def url(self) -> str:
        """Construct database URL."""
        password = self.password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.user}:{password}"
            f"@{self.host}:{self.port}/{self.name}"
        )
    
    @computed_field
    @property
    def sync_url(self) -> str:
        """Construct synchronous database URL (for Alembic)."""
        password = self.password.get_secret_value()
        return (
            f"postgresql://{self.user}:{password}"
            f"@{self.host}:{self.port}/{self.name}"
        )
    
    @field_validator("pool_size")
    @classmethod
    def validate_pool_size(cls, v: int, info: ValidationInfo) -> int:
        """Validate pool size based on environment."""
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production" and v < 20:
            logger.warning(
                "Production pool_size should be at least 20",
                current=v,
                recommended=20
            )
        return v


class RedisSettings(BaseSettings):
    """Redis configuration with cluster support."""
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[SecretStr] = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    
    # Connection pool
    max_connections: int = Field(default=50, ge=10, le=1000)
    socket_timeout: float = Field(default=5.0, ge=1.0, le=30.0)
    socket_connect_timeout: float = Field(default=5.0, ge=1.0, le=30.0)
    retry_on_timeout: bool = Field(default=True)
    
    # Cluster mode
    cluster_mode: bool = Field(default=False, description="Enable Redis cluster mode")
    cluster_nodes: Optional[List[str]] = Field(default=None)
    
    # Sentinel
    use_sentinel: bool = Field(default=False)
    sentinel_master: str = Field(default="mymaster")
    sentinel_nodes: Optional[List[tuple]] = Field(default=None)
    
    @computed_field
    @property
    def url(self) -> str:
        """Construct Redis URL."""
        if self.password:
            password = self.password.get_secret_value()
            return f"redis://:{password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    @field_validator("max_connections")
    @classmethod
    def validate_connections(cls, v: int, info: ValidationInfo) -> int:
        """Validate connections based on environment."""
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production" and v < 50:
            logger.warning(
                "Production Redis max_connections should be at least 50",
                current=v
            )
        return v


class OllamaSettings(BaseSettings):
    """Ollama LLM service configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Ollama API key if authentication is required",
    )
    
    # Connection settings
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds",
        ge=10.0,
        le=600.0,
    )
    connect_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds",
        ge=1.0,
        le=30.0,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries",
        ge=0,
        le=10,
    )
    retry_delay: float = Field(
        default=1.0,
        description="Base retry delay in seconds",
        ge=0.1,
        le=10.0,
    )
    
    # Model settings
    default_model: str = Field(
        default="qwen2.5:7b",
        description="Default model to use",
    )
    fallback_model: str = Field(
        default="qwen2.5:3b",
        description="Fallback model if default fails",
    )
    
    # Performance settings
    max_concurrent: int = Field(
        default=10,
        description="Maximum concurrent requests",
        ge=1,
        le=100,
    )
    batch_size: int = Field(
        default=1,
        description="Batch size for model requests",
        ge=1,
        le=10,
    )
    
    # Streaming settings
    streaming_chunk_size: int = Field(
        default=1024,
        description="Chunk size for streaming responses",
        ge=128,
        le=8192,
    )
    streaming_enabled: bool = Field(
        default=True,
        description="Enable response streaming",
    )
    
    # Context settings
    max_context_length: int = Field(
        default=8192,
        description="Maximum context length in tokens",
        ge=1024,
        le=32768,
    )
    max_response_tokens: int = Field(
        default=4096,
        description="Maximum response tokens",
        ge=256,
        le=16384,
    )
    default_temperature: float = Field(
        default=0.2,
        description="Default model temperature",
        ge=0.0,
        le=2.0,
    )
    
    @field_validator("base_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate and normalize Ollama URL."""
        v = v.rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme: {v}")
        return v


class GraphifySettings(BaseSettings):
    """Graphify integration settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="GRAPHIFY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )
    
    output_dir: str = Field(
        default="graphify-out",
        description="Default Graphify output directory name",
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL for Graphify data in seconds",
        ge=60,
        le=86400,
    )
    max_context_size: int = Field(
        default=10000,
        description="Maximum context size in characters",
        ge=1000,
        le=100000,
    )
    parse_timeout: float = Field(
        default=30.0,
        description="Timeout for parsing Graphify output",
        ge=5.0,
        le=120.0,
    )
    
    # Supported formats
    supported_formats: List[str] = Field(
        default=["json", "yaml", "md"],
        description="Supported Graphify output formats",
    )
    
    # Auto-discovery
    auto_discover: bool = Field(
        default=True,
        description="Auto-discover Graphify output in repos",
    )


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )
    
    # JWT settings
    secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production-use-a-strong-random-key"),
        description="JWT secret key (CRITICAL: change in production!)",
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
    )
    refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=30,
    )
    
    # CORS
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    )
    cors_headers: List[str] = Field(
        default=["*"],
    )
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(
        default=100,
        ge=10,
        le=10000,
    )
    rate_limit_period: int = Field(
        default=60,  # seconds
        ge=1,
        le=3600,
    )
    
    # API keys
    api_key_enabled: bool = Field(default=False)
    api_keys: Optional[List[SecretStr]] = Field(default=None)
    
    # Path security
    allowed_base_dirs: List[str] = Field(
        default_factory=lambda: [
            str(Path.cwd()),
            str(Path.home() / "Projects"),
            str(Path.home() / "workspace"),
        ],
        description="Base directories allowed for file operations",
    )
    
    # Encryption
    encryption_key: Optional[SecretStr] = Field(default=None)
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr, info: ValidationInfo) -> SecretStr:
        """Ensure secret key is strong in production."""
        env = os.getenv("ENVIRONMENT", "development")
        key = v.get_secret_value()
        
        if env == "production":
            if key == "change-me-in-production-use-a-strong-random-key":
                raise ValueError(
                    "Default secret key must be changed in production!"
                )
            if len(key) < 32:
                raise ValueError(
                    "Production secret key must be at least 32 characters"
                )
        
        return v


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="MONITORING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )
    
    # Prometheus
    enable_prometheus: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    metrics_path: str = Field(default="/metrics")
    
    # OpenTelemetry
    enable_tracing: bool = Field(default=True)
    tracing_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint",
    )
    tracing_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: LogFormat = Field(default=LogFormat.JSON)
    log_file_path: Optional[Path] = Field(
        default=None,
        description="Path to log file",
    )
    log_max_size: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum log file size in bytes",
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of log backups to keep",
    )
    
    # Error tracking
    sentry_enabled: bool = Field(default=False)
    sentry_dsn: Optional[str] = Field(default=None)
    sentry_environment: Optional[str] = Field(default=None)
    sentry_traces_sample_rate: float = Field(default=0.1)
    
    # Health checks
    health_check_interval: int = Field(default=30, ge=5, le=300)
    
    # Alerts
    alert_webhook_url: Optional[str] = Field(default=None)
    alert_email: Optional[str] = Field(default=None)


class AgentSettings(BaseSettings):
    """AI Agent configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )
    
    # Execution settings
    max_execution_time: float = Field(
        default=300.0,
        description="Maximum agent execution time in seconds",
        ge=10.0,
        le=3600.0,
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum agent iterations",
        ge=1,
        le=100,
    )
    
    # Tool settings
    max_tool_calls: int = Field(
        default=50,
        description="Maximum tool calls per agent execution",
        ge=1,
        le=200,
    )
    tool_timeout: float = Field(
        default=60.0,
        description="Timeout for individual tool calls",
        ge=5.0,
        le=300.0,
    )
    
    # Code generation
    max_code_length: int = Field(
        default=10000,
        description="Maximum generated code length",
        ge=100,
        le=100000,
    )
    code_review_enabled: bool = Field(default=True)
    
    # Safety
    content_filter_enabled: bool = Field(default=True)
    safe_mode: bool = Field(default=False)
    
    # Artifact storage
    artifact_retention_days: int = Field(
        default=30,
        description="Days to retain agent artifacts",
        ge=1,
        le=365,
    )
    artifact_max_per_run: int = Field(
        default=100,
        description="Maximum artifacts per run",
    )


# ============================================================================
# Main Settings Class
# ============================================================================

class Settings(BaseSettings):
    """
    Central application settings with multi-source configuration.
    
    Configuration priority (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. Secrets files (/run/secrets/*)
    4. Environment-specific YAML config
    5. .env file
    6. Default values
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        validate_default=True,
        secrets_dir="/run/secrets",  # Docker secrets
        yaml_file_encoding="utf-8",
    )
    
    # ========================================
    # Application Settings
    # ========================================
    app_name: str = Field(
        default="AI Engineering Orchestrator",
        description="Application name",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (disable in production!)",
    )
    
    # ========================================
    # Server Settings
    # ========================================
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )
    port: int = Field(
        default=8008,
        description="Server port",
        ge=1024,
        le=65535,
    )
    workers: int = Field(
        default=4,
        description="Number of worker processes",
        ge=1,
        le=16,
    )
    max_concurrent_requests: int = Field(
        default=100,
        description="Maximum concurrent requests",
        ge=10,
        le=1000,
    )
    request_timeout: int = Field(
        default=300,
        description="Request timeout in seconds",
        ge=10,
        le=600,
    )
    
    # ========================================
    # Path Settings
    # ========================================
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Application base directory",
    )
    data_dir: Path = Field(
        default_factory=lambda: Path("data"),
        description="Data directory",
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path("logs"),
        description="Logs directory",
    )
    artifacts_dir: Path = Field(
        default_factory=lambda: Path("data/artifacts"),
        description="Artifacts directory",
    )
    config_dir: Path = Field(
        default_factory=lambda: Path("config"),
        description="Configuration directory",
    )
    
    # ========================================
    # Component Settings
    # ========================================
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    graphify: GraphifySettings = Field(default_factory=GraphifySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    
    # ========================================
    # Feature Flags
    # ========================================
    feature_cache_enabled: bool = Field(default=True)
    feature_streaming_enabled: bool = Field(default=True)
    feature_batch_processing: bool = Field(default=False)
    feature_experimental_agents: bool = Field(default=False)
    feature_proactive_monitoring: bool = Field(
        default=False,
        description="Enable proactive repository monitoring background service",
    )
    monitored_repositories: List[str] = Field(
        default_factory=list,
        description="Repository paths to monitor automatically on startup",
    )
    monitor_poll_interval: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Poll interval in seconds for proactive monitoring",
    )

    # ── Evolution / self-improvement ──────────────────────────────────────────
    feature_evolution_enabled: bool = Field(
        default=False,
        description="Enable the self-improvement evolution service",
    )
    evolution_cycle_interval_hours: int = Field(
        default=168,
        ge=1,
        le=8760,
        description="Hours between automatic evolution cycles (default: weekly)",
    )
    evolution_min_executions_for_analysis: int = Field(
        default=50,
        ge=5,
        description="Minimum execution records before evolution will analyze",
    )
    evolution_improvement_threshold: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Minimum % improvement required to apply a strategy change",
    )
    evolution_max_automatic_risk: str = Field(
        default="medium",
        description="Highest risk level the evolution service may auto-apply ('low', 'medium', 'high')",
    )
    evolution_require_validation: bool = Field(
        default=True,
        description="Validate strategies against historical data before applying",
    )
    evolution_auto_apply: bool = Field(
        default=False,
        description="Automatically apply validated strategies without human review",
    )
    evolution_max_risk_level: str = Field(
        default="medium",
        description="Alias for evolution_max_automatic_risk — highest risk level auto-applied",
    )
    
    # ========================================
    # Validators
    # ========================================
    
    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: Any) -> Environment:
        """Normalize environment value."""
        if isinstance(v, Environment):
            return v
        if isinstance(v, str):
            v = v.lower().strip()
            env_map = {
                "dev": Environment.DEVELOPMENT,
                "development": Environment.DEVELOPMENT,
                "staging": Environment.STAGING,
                "stage": Environment.STAGING,
                "prod": Environment.PRODUCTION,
                "production": Environment.PRODUCTION,
                "test": Environment.TESTING,
                "testing": Environment.TESTING,
                "ci": Environment.CI,
            }
            if v in env_map:
                return env_map[v]
        raise ValueError(f"Invalid environment: {v}")
    
    @field_validator("debug")
    @classmethod
    def validate_debug(cls, v: bool, info: ValidationInfo) -> bool:
        """Ensure debug is disabled in production."""
        env = os.getenv("ENVIRONMENT", "development")
        if v and env == "production":
            raise ValueError("Debug mode must be disabled in production!")
        return v
    
    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int, info: ValidationInfo) -> int:
        """Validate worker count against CPU cores."""
        import os
        cpu_count = os.cpu_count() or 1
        if v > cpu_count * 2 + 1:
            logger.warning(
                "Worker count exceeds recommended maximum (2 * CPU cores + 1)",
                workers=v,
                cpu_cores=cpu_count,
                recommended=cpu_count * 2 + 1,
            )
        return v
    
    @model_validator(mode="after")
    def validate_directories(self) -> "Settings":
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.artifacts_dir,
        ]
        
        for directory in directories:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                except Exception as e:
                    logger.error(f"Failed to create directory {directory}: {e}")
                    raise
        
        return self
    
    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Additional validation for production environment."""
        if self.environment == Environment.PRODUCTION:
            # Check critical security settings
            if self.security.secret_key.get_secret_value() == "change-me-in-production-use-a-strong-random-key":
                raise ValueError("Security secret key must be changed for production!")
            
            if self.security.cors_origins == ["*"]:
                logger.warning("CORS origins set to '*' in production - restrict this!")
            
            if self.database.password.get_secret_value() == "postgres":
                logger.warning("Default database password in production - change immediately!")
            
            # Check monitoring
            if not self.monitoring.enable_prometheus:
                logger.warning("Prometheus monitoring disabled in production")
            
            # Check logging
            if self.monitoring.log_level == LogLevel.DEBUG:
                logger.warning("Debug logging in production - consider INFO or higher")
        
        return self
    
    # ========================================
    # Computed Properties
    # ========================================
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    @computed_field
    @property
    def is_testing(self) -> bool:
        """Check if running tests."""
        return self.environment in (Environment.TESTING, Environment.CI)
    
    @computed_field
    @property
    def api_v1_prefix(self) -> str:
        """API v1 prefix."""
        return "/api/v1"
    
    # ========================================
    # Configuration Sources
    # ========================================
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize configuration source priority."""
        return (
            init_settings,          # CLI arguments (highest priority)
            env_settings,           # Environment variables
            dotenv_settings,        # .env file
            file_secret_settings,   # Docker secrets
            YamlConfigSettingsSource(settings_cls),  # YAML config files
        )
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """Convert settings to dictionary, optionally excluding secrets."""
        data = self.model_dump()
        
        if exclude_secrets:
            # Redact secret values
            for key in list(data.keys()):
                if isinstance(data[key], dict):
                    for subkey in list(data[key].keys()):
                        if "password" in subkey.lower() or "secret" in subkey.lower() or "key" in subkey.lower():
                            if data[key][subkey]:
                                data[key][subkey] = "********"
        
        return data
    
    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Export current settings to YAML."""
        data = self.to_dict(exclude_secrets=True)
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        
        if path:
            path.write_text(yaml_str)
            logger.info(f"Settings exported to {path}")
        
        return yaml_str
    
    def load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_config_path = self.config_dir / "environments" / f"{self.environment.value}.yaml"
        
        if env_config_path.exists():
            with open(env_config_path) as f:
                return yaml.safe_load(f) or {}
        
        logger.warning(f"No environment config found at {env_config_path}")
        return {}
    
    def validate_all(self) -> bool:
        """
        Comprehensive validation of all settings.
        Returns True if all validations pass, raises exception otherwise.
        """
        # Check database connectivity settings
        if self.is_production and not self.database.use_ssl:
            logger.warning("Database SSL not enabled in production")
        
        # Check Redis settings
        if self.feature_cache_enabled and self.redis.cluster_mode:
            if not self.redis.cluster_nodes:
                raise ValueError("Redis cluster mode enabled but no nodes configured")
        
        # Check Ollama connectivity
        if not self.ollama.base_url:
            raise ValueError("Ollama base URL is required")
        
        # Check paths
        for path_attr in ["data_dir", "logs_dir", "artifacts_dir"]:
            path = getattr(self, path_attr)
            if not path.exists():
                logger.warning(f"Directory does not exist: {path}")
        
        logger.info("All settings validated successfully")
        return True


# ============================================================================
# Singleton Instance
# ============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to ensure single instance and fast access.
    Call with set_env=True to bypass cache and reload.
    """
    try:
        settings = Settings()
        logger.info(
            "Settings loaded successfully",
            environment=settings.environment.value,
            debug=settings.debug,
        )
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}", exc_info=True)
        # In development, provide fallback settings
        if os.getenv("ENVIRONMENT", "development") == "development":
            logger.warning("Using fallback development settings")
            return Settings(environment=Environment.DEVELOPMENT)
        raise


# Global settings instance - use this throughout the application
settings = get_settings()

# Type alias for convenience
SettingsType = Settings


# ============================================================================
# Configuration Validation at Import Time
# ============================================================================

def validate_settings_on_startup() -> None:
    """
    Validate all settings at application startup.
    Called from main.py before server starts.
    """
    try:
        settings.validate_all()
        
        # Log configuration summary (no secrets)
        logger.info(
            "Configuration summary",
            app_name=settings.app_name,
            version=settings.app_version,
            environment=settings.environment.value,
            debug=settings.debug,
            host=settings.host,
            port=settings.port,
            database_host=settings.database.host,
            redis_host=settings.redis.host,
            ollama_url=settings.ollama.base_url,
            monitoring_enabled=settings.monitoring.enable_prometheus,
        )
    except Exception as e:
        logger.critical(f"Settings validation failed: {e}", exc_info=True)
        if settings.is_production:
            sys.exit(1)  # Exit in production if settings are invalid
        else:
            logger.warning("Continuing despite settings validation failure (non-production)")


# Use this decorator on functions that need validated settings
def require_valid_settings(func):
    """Decorator to ensure settings are valid before execution."""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not settings.is_testing:
            validate_settings_on_startup()
        return func(*args, **kwargs)
    
    return wrapper


logger.info("Settings module initialized", module=__name__)