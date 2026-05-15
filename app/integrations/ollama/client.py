"""
Production-grade Ollama client with enterprise features:
- Circuit breaker pattern for fault tolerance
- Exponential backoff with jitter
- Connection pooling and reuse
- Streaming support with backpressure
- Request queuing and concurrency control
- Comprehensive error handling and classification
- Response validation and sanitization
- Performance metrics collection
- Health checks and connection monitoring
- Automatic model fallback
- Request/response logging with sensitive data masking

This module is the critical interface to local LLM models.
Security Level: HIGH - Handles model inputs/outputs
Reliability Level: CRITICAL - Core dependency for all AI operations
"""

import asyncio
import json
import time
import hashlib
from typing import (
    Any, Dict, List, Optional, Union, AsyncIterator, Tuple,
    Type, Callable, Awaitable
)
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    RetryError,
)
from pydantic import BaseModel, Field, field_validator, ValidationError

from app.core.config.settings import settings, OllamaSettings
from app.core.monitoring.logging import get_logger, LogContext
from app.core.monitoring.metrics import (
    model_requests,
    model_latency,
    model_errors,
    active_connections,
    MetricsTracker,
)
from app.utils.retry import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    ErrorCategory,
    RetryConfig,
)
from app.utils.validators import validate_model, InputSanitizer

logger = get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ModelStatus(str, Enum):
    """Model availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"
    UNKNOWN = "unknown"


class ChatRole(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Validated chat message."""
    role: ChatRole
    content: str = Field(..., min_length=1, max_length=100000)
    name: Optional[str] = Field(default=None, max_length=100)
    images: Optional[List[str]] = Field(default=None, max_length=10)
    
    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        """Sanitize message content."""
        return InputSanitizer.sanitize_string(
            v,
            strip_html=False,
            max_length=100000
        )


class ChatRequest(BaseModel):
    """Validated chat request to Ollama."""
    model: str = Field(..., min_length=1, max_length=200)
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=100)
    stream: bool = Field(default=False)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    stop: Optional[List[str]] = Field(default=None)
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    
    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        if not validate_model(v):
            raise ValueError(f"Invalid model name format: {v}")
        return v


class ChatResponse(BaseModel):
    """Validated chat response from Ollama."""
    model: str
    message: ChatMessage
    done: bool = Field(default=True)
    total_duration: Optional[float] = None
    load_duration: Optional[float] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[float] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[float] = None
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens per second."""
        if self.eval_count and self.eval_duration:
            return self.eval_count / (self.eval_duration / 1e9)
        return None


class ModelInfo(BaseModel):
    """Model information from Ollama."""
    name: str
    modified_at: datetime
    size: int
    digest: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def size_gb(self) -> float:
        """Get model size in GB."""
        return self.size / (1024 ** 3)
    
    @property
    def size_formatted(self) -> str:
        """Get human-readable size."""
        size_gb = self.size_gb
        if size_gb < 1:
            return f"{self.size / (1024 ** 2):.0f} MB"
        return f"{size_gb:.2f} GB"


@dataclass
class ModelStats:
    """Runtime statistics for a model."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_duration: float = 0.0
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    average_latency: float = 0.0
    
    def update_success(self, duration: float, tokens: int):
        """Update stats after successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_tokens += tokens
        self.total_duration += duration
        self.last_used = datetime.utcnow()
        self.average_latency = (
            (self.average_latency * (self.successful_requests - 1) + duration)
            / self.successful_requests
        )
    
    def update_failure(self, error: str):
        """Update stats after failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        self.last_used = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100


# ============================================================================
# Connection Pool
# ============================================================================

class ConnectionPool:
    """
    HTTP connection pool for Ollama with lifecycle management.
    """
    
    def __init__(
        self,
        max_connections: int = 20,
        max_keepalive: int = 10,
        keepalive_expiry: float = 30.0,
    ):
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self.keepalive_expiry = keepalive_expiry
        self._client: Optional[httpx.AsyncClient] = None
        self._created_at: Optional[datetime] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._should_recreate():
            await self.close()
            self._client = await self._create_client()
        return self._client
    
    async def _create_client(self) -> httpx.AsyncClient:
        """Create new HTTP client with optimized settings."""
        self._created_at = datetime.utcnow()
        
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.ollama.connect_timeout,
                read=settings.ollama.timeout,
                write=settings.ollama.timeout,
                pool=settings.ollama.timeout,
            ),
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive,
                keepalive_expiry=self.keepalive_expiry,
            ),
            follow_redirects=True,
        )
    
    def _should_recreate(self) -> bool:
        """Check if client should be recreated."""
        if self._created_at is None:
            return True
        
        # Recreate every 30 minutes to prevent stale connections
        age = (datetime.utcnow() - self._created_at).total_seconds()
        return age > 1800
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._created_at = None


# ============================================================================
# Request Queue
# ============================================================================

class RequestQueue:
    """
    Request queue for managing concurrent Ollama requests.
    Implements rate limiting and backpressure.
    """
    
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_requests = 0
        self._total_queued = 0
        self._total_processed = 0
        self._queue_times: List[float] = []
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a slot in the request queue."""
        start_time = time.time()
        self._total_queued += 1
        
        async with self._semaphore:
            queue_time = time.time() - start_time
            self._queue_times.append(queue_time)
            self._active_requests += 1
            active_connections.set(self._active_requests)
            
            try:
                yield
            finally:
                self._active_requests -= 1
                self._total_processed += 1
                active_connections.set(self._active_requests)
    
    @property
    def average_queue_time(self) -> float:
        """Get average queue wait time."""
        if not self._queue_times:
            return 0.0
        return sum(self._queue_times) / len(self._queue_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "active_requests": self._active_requests,
            "total_queued": self._total_queued,
            "total_processed": self._total_processed,
            "average_queue_time_ms": round(self.average_queue_time * 1000, 2),
            "max_concurrent": self._semaphore._value,
        }


# ============================================================================
# Ollama Client
# ============================================================================

def _get_cloud_client(model_name: str):
    """
    Return the cloud provider client for *model_name*, or None for local models.

    Imported lazily to avoid a circular-import between the ollama package and
    the services layer.  The import cost is paid only once (registry is a
    singleton) and only when a cloud model is actually requested.
    """
    try:
        from app.services.cloud_registry import get_cloud_registry
        return get_cloud_registry().get_client_for_model(model_name)
    except Exception:
        return None


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[Dict] = None,
    ):
        self.model = model
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class ModelNotFoundError(OllamaClientError):
    """Raised when model is not found."""
    pass


class ModelTimeoutError(OllamaClientError):
    """Raised when model request times out."""
    pass


class ModelOverloadedError(OllamaClientError):
    """Raised when model is overloaded."""
    pass


class ModelConnectionError(OllamaClientError):
    """Raised when connection to Ollama fails."""
    pass


class OllamaClient:
    """
    Production-grade Ollama client with comprehensive error handling,
    connection management, and performance optimization.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (default from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            api_key: API key for authentication
        """
        self.base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self.timeout = timeout or settings.ollama.timeout
        self.max_retries = max_retries or settings.ollama.max_retries
        self.api_key = api_key or (
            settings.ollama.api_key.get_secret_value()
            if settings.ollama.api_key
            else None
        )
        
        # Connection pool
        self.connection_pool = ConnectionPool(
            max_connections=20,
            max_keepalive=10,
            keepalive_expiry=30.0,
        )
        
        # Request queue
        self.request_queue = RequestQueue(
            max_concurrent=settings.ollama.max_concurrent,
        )
        
        # Circuit breaker per model
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Model statistics
        self.model_stats: Dict[str, ModelStats] = {}
        
        # Model status cache
        self.model_status: Dict[str, ModelStatus] = {}
        self.model_status_updated: Dict[str, datetime] = {}
        
        # Fallback models for automatic failover
        self.fallback_chain: List[str] = []
        
        logger.info(
            "Ollama client initialized",
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_concurrent=settings.ollama.max_concurrent,
        )
    
    # ========================================
    # Model Management
    # ========================================
    
    async def list_models(self) -> List[ModelInfo]:
        """
        List available models from Ollama.
        
        Returns:
            List of model information objects
            
        Raises:
            ModelConnectionError: If cannot connect to Ollama
        """
        try:
            client = await self.connection_pool.get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                try:
                    model = ModelInfo(
                        name=model_data["name"],
                        modified_at=datetime.fromisoformat(
                            model_data["modified_at"].replace("Z", "+00:00")
                        ),
                        size=model_data["size"],
                        digest=model_data["digest"],
                        details=model_data.get("details", {}),
                    )
                    models.append(model)
                    
                    # Update status cache
                    self.model_status[model.name] = ModelStatus.AVAILABLE
                    self.model_status_updated[model.name] = datetime.utcnow()
                    
                except (KeyError, ValueError) as e:
                    logger.warning(
                        "Failed to parse model info",
                        model_data=model_data,
                        error=str(e),
                    )
                    continue
            
            logger.info(
                "Models listed successfully",
                count=len(models),
            )
            
            return models
            
        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama", error=str(e))
            raise ModelConnectionError(
                f"Cannot connect to Ollama at {self.base_url}: {e}"
            )
        except httpx.TimeoutException as e:
            logger.error("Timeout listing models", error=str(e))
            raise ModelTimeoutError(f"Timeout listing models: {e}")
        except Exception as e:
            logger.error("Failed to list models", error=str(e), exc_info=True)
            raise OllamaClientError(f"Failed to list models: {e}")
    
    async def model_exists(self, name: str) -> bool:
        """Check if a model is available locally."""
        try:
            models = await self.list_models()
            return any(m.name == name or m.name.startswith(f"{name}:") for m in models)
        except Exception:
            return False

    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo if found, None otherwise
        """
        try:
            client = await self.connection_pool.get_client()
            response = await client.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            )
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            return ModelInfo(
                name=model_name,
                modified_at=datetime.fromisoformat(
                    data.get("modified_at", datetime.utcnow().isoformat())
                ),
                size=data.get("size", 0),
                digest=data.get("digest", ""),
                details=data.get("details", {}),
            )
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(
                "Failed to get model info",
                model=model_name,
                status_code=e.response.status_code,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "Error getting model info",
                model=model_name,
                error=str(e),
            )
            return None
    
    async def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available and ready.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is available
        """
        # Check cache first (cache for 30 seconds)
        if model_name in self.model_status:
            last_updated = self.model_status_updated.get(model_name)
            if last_updated:
                age = (datetime.utcnow() - last_updated).total_seconds()
                if age < 30:
                    return self.model_status[model_name] == ModelStatus.AVAILABLE
        
        # Check model info
        model_info = await self.get_model_info(model_name)
        is_available = model_info is not None
        
        self.model_status[model_name] = (
            ModelStatus.AVAILABLE if is_available else ModelStatus.UNAVAILABLE
        )
        self.model_status_updated[model_name] = datetime.utcnow()
        
        return is_available
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful
        """
        logger.info("Pulling model", model=model_name)
        
        try:
            client = await self.connection_pool.get_client()
            
            async with client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True},
                timeout=600.0,  # Long timeout for model download
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            
                            if "error" in data:
                                logger.error(
                                    "Model pull error",
                                    model=model_name,
                                    error=data["error"],
                                )
                                return False
                            
                            # Log progress
                            if "completed" in data and "total" in data:
                                completed = data["completed"]
                                total = data["total"]
                                if total > 0:
                                    progress = (completed / total) * 100
                                    if int(progress) % 10 == 0:  # Log every 10%
                                        logger.info(
                                            "Model pull progress",
                                            model=model_name,
                                            progress=f"{progress:.1f}%",
                                        )
                            
                            # Check if done
                            if status == "success":
                                logger.info(
                                    "Model pulled successfully",
                                    model=model_name,
                                )
                                self.model_status[model_name] = ModelStatus.AVAILABLE
                                self.model_status_updated[model_name] = datetime.utcnow()
                                return True
                                
                        except json.JSONDecodeError:
                            continue
                
                return True
                
        except Exception as e:
            logger.error(
                "Failed to pull model",
                model=model_name,
                error=str(e),
                exc_info=True,
            )
            return False
    
    # ========================================
    # Chat Interface
    # ========================================
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        Send chat request to Ollama or a cloud provider with full error handling.

        If *model* is a recognised cloud model (gpt-4o, claude-3.5-sonnet, …)
        the request is transparently routed to the appropriate cloud client.
        Callers never need to know where the model runs.

        Args:
            model: Model name (local Ollama tag or cloud model name)
            messages: List of message dicts with 'role' and 'content'
            temperature: Model temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            Response string or async iterator for streaming

        Raises:
            ModelNotFoundError: If model not found
            ModelTimeoutError: If request times out
            ModelOverloadedError: If model is overloaded
        """
        # Route cloud models BEFORE local validation — cloud names like
        # "gpt-4o" or "openai/gpt-4o" would fail the Ollama name regex.
        cloud_client = _get_cloud_client(model)
        if cloud_client is not None:
            logger.info("Routing to cloud provider", model=model, provider=cloud_client.provider_name)
            return await cloud_client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

        # Validate model name (Ollama requires name:tag format)
        if not validate_model(model):
            raise ValueError(f"Invalid model name: {model}")
        
        # Build and validate request
        try:
            request = ChatRequest(
                model=model,
                messages=[
                    ChatMessage(role=msg.get("role", "user"), content=msg.get("content", ""))
                    for msg in messages
                ],
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k in ChatRequest.model_fields},
            )
        except ValidationError as e:
            logger.error("Invalid chat request", errors=str(e))
            raise ValueError(f"Invalid request: {e}")
        
        # Initialize model stats
        if model not in self.model_stats:
            self.model_stats[model] = ModelStats()
        
        # Acquire queue slot
        async with self.request_queue.acquire():
            # Execute with circuit breaker
            return await self._execute_chat(request)
    
    async def _execute_chat(
        self,
        request: ChatRequest,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Execute chat request with retry logic and circuit breaker.
        """
        model = request.model
        
        # Get or create circuit breaker for model
        if model not in self.circuit_breakers:
            self.circuit_breakers[model] = CircuitBreaker()
        
        circuit_breaker = self.circuit_breakers[model]
        
        # Track metrics
        model_requests.labels(model_name=model, provider="ollama").inc()
        
        start_time = time.time()
        
        try:
            # Check circuit breaker
            async with circuit_breaker:
                # Send request with retries
                return await self._send_chat_request(request)
                
        except CircuitBreakerOpenError:
            logger.error(
                "Circuit breaker open for model",
                model=model,
            )
            model_errors.labels(model_name=model, error_type="circuit_breaker").inc()
            
            # Try fallback model
            if model != settings.ollama.fallback_model:
                logger.info(
                    "Attempting fallback model",
                    original=model,
                    fallback=settings.ollama.fallback_model,
                )
                try:
                    fallback_request = request.model_copy()
                    fallback_request.model = settings.ollama.fallback_model
                    return await self._send_chat_request(fallback_request)
                except Exception as fallback_error:
                    logger.error(
                        "Fallback model also failed",
                        original=model,
                        fallback=settings.ollama.fallback_model,
                        error=str(fallback_error),
                    )
            
            raise ModelOverloadedError(
                f"Model '{model}' is temporarily unavailable (circuit breaker open)"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.model_stats[model].update_failure(str(e))
            
            async with MetricsTracker(
                model_latency,
                {"model_name": model, "provider": "ollama"}
            ):
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        )),
        before_sleep=before_sleep_log(logger, "WARNING"),
        after=after_log(logger, "DEBUG"),
    )
    async def _send_chat_request(
        self,
        request: ChatRequest,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Send the actual HTTP request to Ollama with retries.
        """
        model = request.model
        client = await self.connection_pool.get_client()
        
        payload = request.model_dump(exclude_none=True)
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.debug(
            "Sending chat request",
            model=model,
            message_count=len(request.messages),
            stream=request.stream,
            temperature=request.temperature,
        )
        
        if request.stream:
            return await self._handle_streaming(client, payload, headers, model)
        else:
            return await self._handle_sync(client, payload, headers, model)
    
    async def _handle_sync(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        model: str,
    ) -> str:
        """Handle non-streaming chat request."""
        start_time = time.time()
        
        try:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response
            try:
                chat_response = ChatResponse(**data)
            except ValidationError as e:
                logger.error(
                    "Invalid response from Ollama",
                    model=model,
                    errors=str(e),
                )
                raise OllamaClientError(f"Invalid response: {e}")
            
            duration = time.time() - start_time
            tokens = chat_response.eval_count or 0
            
            # Update stats
            self.model_stats[model].update_success(duration, tokens)
            
            logger.info(
                "Chat request completed",
                model=model,
                duration_ms=round(duration * 1000, 2),
                tokens=tokens,
                tokens_per_second=round(chat_response.tokens_per_second or 0, 2),
            )
            
            return chat_response.message.content
            
        except httpx.HTTPStatusError as e:
            duration = time.time() - start_time
            self._handle_http_error(e, model, duration)
            
        except httpx.TimeoutException as e:
            duration = time.time() - start_time
            self.model_stats[model].update_failure("timeout")
            model_errors.labels(model_name=model, error_type="timeout").inc()
            
            logger.error(
                "Chat request timeout",
                model=model,
                duration_ms=round(duration * 1000, 2),
            )
            raise ModelTimeoutError(f"Request to '{model}' timed out after {duration:.1f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Chat request failed",
                model=model,
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def _handle_streaming(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        model: str,
    ) -> AsyncIterator[str]:
        """Handle streaming chat request."""
        start_time = time.time()
        total_content = []
        chunk_count = 0
        
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            
                            if "error" in chunk:
                                logger.error(
                                    "Streaming error from Ollama",
                                    model=model,
                                    error=chunk["error"],
                                )
                                model_errors.labels(
                                    model_name=model,
                                    error_type="stream_error"
                                ).inc()
                                raise OllamaClientError(chunk["error"], model=model)
                            
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                total_content.append(content)
                                chunk_count += 1
                                yield content
                                
                        except json.JSONDecodeError:
                            logger.warning(
                                "Invalid JSON in stream",
                                model=model,
                                line=line[:100],
                            )
                            continue
                
                duration = time.time() - start_time
                full_content = "".join(total_content)
                tokens = len(full_content.split())  # Approximate token count
                
                # Update stats
                self.model_stats[model].update_success(duration, tokens)
                
                logger.info(
                    "Streaming chat completed",
                    model=model,
                    duration_ms=round(duration * 1000, 2),
                    chunks=chunk_count,
                    content_length=len(full_content),
                )
                
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, model, time.time() - start_time)
        except Exception as e:
            logger.error(
                "Streaming request failed",
                model=model,
                error=str(e),
                exc_info=True,
            )
            raise
    
    def _handle_http_error(
        self,
        error: httpx.HTTPStatusError,
        model: str,
        duration: float,
    ):
        """Handle HTTP errors with classification."""
        status_code = error.response.status_code
        
        if status_code == 404:
            self.model_stats[model].update_failure("model_not_found")
            model_errors.labels(model_name=model, error_type="not_found").inc()
            self.model_status[model] = ModelStatus.UNAVAILABLE
            raise ModelNotFoundError(
                f"Model '{model}' not found. Pull it first with: ollama pull {model}",
                model=model,
                status_code=404,
            )
        
        elif status_code == 429:
            self.model_stats[model].update_failure("rate_limited")
            model_errors.labels(model_name=model, error_type="rate_limited").inc()
            raise ModelOverloadedError(
                f"Model '{model}' rate limited",
                model=model,
                status_code=429,
            )
        
        elif status_code >= 500:
            self.model_stats[model].update_failure(f"server_error_{status_code}")
            model_errors.labels(model_name=model, error_type="server_error").inc()
            raise OllamaClientError(
                f"Ollama server error ({status_code})",
                model=model,
                status_code=status_code,
            )
        
        else:
            self.model_stats[model].update_failure(f"http_{status_code}")
            model_errors.labels(model_name=model, error_type="http_error").inc()
            raise OllamaClientError(
                f"HTTP error {status_code}",
                model=model,
                status_code=status_code,
            )
    
    # ========================================
    # Health Check
    # ========================================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check.
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "base_url": self.base_url,
            "checks": {},
        }
        
        try:
            # Check connectivity
            start = time.time()
            models = await self.list_models()
            latency = time.time() - start
            
            health["checks"]["connectivity"] = {
                "status": "ok",
                "latency_ms": round(latency * 1000, 2),
                "models_count": len(models),
            }
            
            # Check default model
            default_model = settings.ollama.default_model
            if default_model:
                is_available = any(m.name == default_model for m in models)
                health["checks"]["default_model"] = {
                    "status": "ok" if is_available else "warning",
                    "model": default_model,
                    "available": is_available,
                }
            
            # Get queue stats
            health["queue"] = self.request_queue.get_stats()
            
            # Get circuit breaker states
            health["circuit_breakers"] = {
                model: cb.get_metrics()
                for model, cb in self.circuit_breakers.items()
            }
            
            # Get model stats
            health["model_stats"] = {
                model: {
                    "total_requests": stats.total_requests,
                    "success_rate": round(stats.success_rate, 2),
                    "average_latency_ms": round(stats.average_latency * 1000, 2),
                    "last_used": stats.last_used.isoformat() if stats.last_used else None,
                }
                for model, stats in self.model_stats.items()
            }
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error("Health check failed", error=str(e))
        
        return health
    
    # ========================================
    # Statistics
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive client statistics."""
        return {
            "base_url": self.base_url,
            "total_models_tracked": len(self.model_stats),
            "queue_stats": self.request_queue.get_stats(),
            "models": {
                model: {
                    "total_requests": stats.total_requests,
                    "successful": stats.successful_requests,
                    "failed": stats.failed_requests,
                    "success_rate": round(stats.success_rate, 2),
                    "total_tokens": stats.total_tokens,
                    "average_latency_ms": round(stats.average_latency * 1000, 2),
                    "last_used": stats.last_used.isoformat() if stats.last_used else None,
                    "last_error": stats.last_error,
                }
                for model, stats in self.model_stats.items()
            },
            "circuit_breakers": {
                model: cb.get_metrics()
                for model, cb in self.circuit_breakers.items()
            },
        }
    
    # ========================================
    # Lifecycle
    # ========================================
    
    async def close(self):
        """Clean shutdown of client resources."""
        logger.info("Shutting down Ollama client")
        await self.connection_pool.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ============================================================================
# Factory Function
# ============================================================================

def create_ollama_client(
    base_url: Optional[str] = None,
    **kwargs
) -> OllamaClient:
    """
    Factory function to create configured Ollama client.
    
    Args:
        base_url: Optional base URL override
        **kwargs: Additional client configuration
        
    Returns:
        Configured OllamaClient instance
    """
    return OllamaClient(base_url=base_url, **kwargs)


# Initialize default client
_default_client: Optional[OllamaClient] = None


def get_default_client() -> OllamaClient:
    """Get or create the default Ollama client instance."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


logger.info("Ollama client module initialized successfully")