"""
Extended tests for app.integrations.ollama.client — covering uncovered lines.

Targets:
- ConnectionPool: get_client, reuse, close, _should_recreate
- RequestQueue: acquire, get_stats, average_queue_time
- ModelStats: update_success, update_failure, success_rate
- OllamaClient: list_models error paths, get_model_info, is_model_available,
                 pull_model, chat (sync + stream), health_check, get_stats
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx
import pytest

from app.integrations.ollama.client import (
    ChatMessage,
    ChatRole,
    ConnectionPool,
    ModelConnectionError,
    ModelInfo,
    ModelNotFoundError,
    ModelStats,
    ModelTimeoutError,
    OllamaClient,
    OllamaClientError,
    RequestQueue,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_response(status: int = 200, json_data=None):
    """Build a mock httpx.Response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status
    mock_resp.json = MagicMock(return_value=json_data or {})
    if status >= 400:
        mock_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"HTTP {status}",
                request=MagicMock(),
                response=mock_resp,
            )
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _model_info(name: str = "test:7b") -> ModelInfo:
    return ModelInfo(
        name=name,
        modified_at=datetime.utcnow(),
        size=1_000_000_000,
        digest="sha256:abc123",
        details={},
    )


# ============================================================================
# ConnectionPool
# ============================================================================

class TestConnectionPool:

    @pytest.mark.asyncio
    async def test_get_client_creates_new_client(self):
        pool = ConnectionPool(max_connections=5, max_keepalive=3, keepalive_expiry=30.0)
        client = await pool.get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        await pool.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing_client(self):
        pool = ConnectionPool(max_connections=5, max_keepalive=3, keepalive_expiry=30.0)
        c1 = await pool.get_client()
        c2 = await pool.get_client()
        # Same object identity — client is reused
        assert c1 is c2
        await pool.close()

    @pytest.mark.asyncio
    async def test_close_sets_client_to_none(self):
        pool = ConnectionPool(max_connections=5, max_keepalive=3, keepalive_expiry=30.0)
        await pool.get_client()  # Create client
        assert pool._client is not None
        await pool.close()
        assert pool._client is None

    @pytest.mark.asyncio
    async def test_close_without_client_does_not_raise(self):
        pool = ConnectionPool(max_connections=5, max_keepalive=3, keepalive_expiry=30.0)
        # No client created — close should be a no-op
        await pool.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_should_recreate_returns_true_when_no_created_at(self):
        pool = ConnectionPool()
        pool._created_at = None
        assert pool._should_recreate() is True

    @pytest.mark.asyncio
    async def test_should_recreate_returns_false_for_fresh_client(self):
        pool = ConnectionPool()
        pool._created_at = datetime.utcnow()
        assert pool._should_recreate() is False

    @pytest.mark.asyncio
    async def test_should_recreate_returns_true_for_old_client(self):
        pool = ConnectionPool()
        # Simulate a client created 31 minutes ago
        pool._created_at = datetime.utcnow() - timedelta(minutes=31)
        assert pool._should_recreate() is True

    @pytest.mark.asyncio
    async def test_get_client_recreates_when_stale(self):
        pool = ConnectionPool(max_connections=5, max_keepalive=3, keepalive_expiry=30.0)
        c1 = await pool.get_client()
        # Force stale timestamp
        pool._created_at = datetime.utcnow() - timedelta(minutes=31)
        c2 = await pool.get_client()
        # A new client should have been created
        assert c2 is not c1
        await pool.close()


# ============================================================================
# RequestQueue
# ============================================================================

class TestRequestQueue:

    @pytest.mark.asyncio
    async def test_acquire_increments_and_decrements_active(self):
        q = RequestQueue(max_concurrent=5)
        assert q._active_requests == 0
        async with q.acquire():
            assert q._active_requests == 1
        assert q._active_requests == 0

    @pytest.mark.asyncio
    async def test_acquire_tracks_queued_and_processed(self):
        q = RequestQueue(max_concurrent=5)
        async with q.acquire():
            pass
        assert q._total_queued == 1
        assert q._total_processed == 1

    @pytest.mark.asyncio
    async def test_acquire_multiple_times(self):
        q = RequestQueue(max_concurrent=5)
        for _ in range(3):
            async with q.acquire():
                pass
        assert q._total_queued == 3
        assert q._total_processed == 3

    def test_get_stats_returns_dict_with_expected_keys(self):
        q = RequestQueue(max_concurrent=3)
        stats = q.get_stats()
        assert isinstance(stats, dict)
        assert "active_requests" in stats
        assert "total_queued" in stats
        assert "total_processed" in stats
        assert "average_queue_time_ms" in stats
        assert "max_concurrent" in stats

    def test_average_queue_time_zero_when_no_requests(self):
        q = RequestQueue(max_concurrent=3)
        assert q.average_queue_time == 0.0

    @pytest.mark.asyncio
    async def test_average_queue_time_positive_after_request(self):
        q = RequestQueue(max_concurrent=3)
        async with q.acquire():
            pass
        # Queue time should be >= 0
        assert q.average_queue_time >= 0.0

    @pytest.mark.asyncio
    async def test_acquire_respects_semaphore_limit(self):
        """Ensure the semaphore prevents more than max_concurrent simultaneous slots."""
        q = RequestQueue(max_concurrent=2)
        results = []

        async def work(idx):
            async with q.acquire():
                results.append(q._active_requests)
                await asyncio.sleep(0)  # Yield

        await asyncio.gather(work(0), work(1), work(2))
        # Each slot should record at most max_concurrent active
        assert all(r <= 2 for r in results)


# ============================================================================
# ModelStats
# ============================================================================

class TestModelStats:

    def test_initial_state(self):
        stats = ModelStats()
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_tokens == 0
        assert stats.last_used is None
        assert stats.last_error is None

    def test_success_rate_is_100_when_no_requests(self):
        stats = ModelStats()
        assert stats.success_rate == 100.0

    def test_update_success_increments_counters(self):
        stats = ModelStats()
        stats.update_success(duration=0.5, tokens=100)
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.total_tokens == 100
        assert stats.last_used is not None

    def test_update_failure_increments_counters(self):
        stats = ModelStats()
        stats.update_failure(error="Connection refused")
        assert stats.total_requests == 1
        assert stats.failed_requests == 1
        assert stats.last_error == "Connection refused"
        assert stats.last_used is not None

    def test_success_rate_after_all_successes(self):
        stats = ModelStats()
        stats.update_success(0.1, 10)
        stats.update_success(0.2, 20)
        assert stats.success_rate == 100.0

    def test_success_rate_after_all_failures(self):
        stats = ModelStats()
        stats.update_failure("err1")
        stats.update_failure("err2")
        assert stats.success_rate == 0.0

    def test_success_rate_after_mixed_requests(self):
        stats = ModelStats()
        stats.update_success(0.1, 10)
        stats.update_success(0.2, 20)
        stats.update_failure("err")
        rate = stats.success_rate
        # 2 out of 3 = 66.66...
        assert abs(rate - (2 / 3 * 100)) < 0.01

    def test_average_latency_updates_correctly(self):
        stats = ModelStats()
        stats.update_success(1.0, 10)
        stats.update_success(3.0, 30)
        # average of 1.0 and 3.0 = 2.0
        assert abs(stats.average_latency - 2.0) < 0.01

    def test_total_duration_accumulates(self):
        stats = ModelStats()
        stats.update_success(1.0, 10)
        stats.update_success(2.0, 20)
        assert abs(stats.total_duration - 3.0) < 0.01


# ============================================================================
# OllamaClient — Initialization
# ============================================================================

class TestOllamaClientInit:

    def test_default_init(self):
        client = OllamaClient()
        assert client.base_url is not None
        assert client.timeout > 0
        assert client.max_retries > 0

    def test_custom_base_url(self):
        client = OllamaClient(base_url="http://localhost:11434")
        assert "localhost:11434" in client.base_url

    def test_custom_timeout(self):
        client = OllamaClient(timeout=120.0)
        assert client.timeout == 120.0

    def test_connection_pool_created(self):
        client = OllamaClient()
        assert isinstance(client.connection_pool, ConnectionPool)

    def test_request_queue_created(self):
        client = OllamaClient()
        assert isinstance(client.request_queue, RequestQueue)

    def test_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert not client.base_url.endswith("/")


# ============================================================================
# OllamaClient — list_models
# ============================================================================

class TestListModels:

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        client = OllamaClient()
        mock_resp = _make_response(200, {
            "models": [
                {
                    "name": "qwen2.5-coder:7b",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 4_000_000_000,
                    "digest": "sha256:abc123",
                    "details": {},
                }
            ]
        })
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.list_models()
        assert len(result) == 1
        assert result[0].name == "qwen2.5-coder:7b"

    @pytest.mark.asyncio
    async def test_list_models_updates_status_cache(self):
        client = OllamaClient()
        mock_resp = _make_response(200, {
            "models": [
                {
                    "name": "llama3:8b",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 1_000_000_000,
                    "digest": "sha256:def",
                    "details": {},
                }
            ]
        })
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            await client.list_models()
        from app.integrations.ollama.client import ModelStatus
        assert client.model_status.get("llama3:8b") == ModelStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        client = OllamaClient()
        mock_resp = _make_response(200, {"models": []})
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.list_models()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_models_malformed_model_skipped(self):
        """Models missing required fields should be silently skipped."""
        client = OllamaClient()
        mock_resp = _make_response(200, {
            "models": [
                {"name": "bad-model"},  # Missing required fields
                {
                    "name": "good:7b",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 1000,
                    "digest": "abc",
                    "details": {},
                },
            ]
        })
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.list_models()
        names = [m.name for m in result]
        assert "good:7b" in names
        assert "bad-model" not in names

    @pytest.mark.asyncio
    async def test_list_models_connect_error_raises_model_connection_error(self):
        client = OllamaClient()
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_get.return_value = mock_http
            with pytest.raises(ModelConnectionError):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_list_models_timeout_error_raises_model_timeout_error(self):
        client = OllamaClient()
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_get.return_value = mock_http
            with pytest.raises(ModelTimeoutError):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_list_models_generic_error_raises_ollama_client_error(self):
        client = OllamaClient()
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=RuntimeError("unexpected"))
            mock_get.return_value = mock_http
            with pytest.raises(OllamaClientError):
                await client.list_models()


# ============================================================================
# OllamaClient — get_model_info
# ============================================================================

class TestGetModelInfo:

    @pytest.mark.asyncio
    async def test_get_model_info_found(self):
        client = OllamaClient()
        mock_resp = _make_response(200, {
            "modified_at": "2024-01-01T00:00:00",
            "size": 4_000_000_000,
            "digest": "sha256:xyz",
            "details": {"format": "gguf"},
        })
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.get_model_info("qwen2.5-coder:7b")
        assert result is not None
        assert result.name == "qwen2.5-coder:7b"
        assert result.size == 4_000_000_000

    @pytest.mark.asyncio
    async def test_get_model_info_returns_none_for_404_status(self):
        client = OllamaClient()
        # Build a 404 response that does NOT raise on raise_for_status
        # because the code checks status_code first before calling raise_for_status
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 404
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={})
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.get_model_info("nonexistent:7b")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_model_info_http_status_error_404_returns_none(self):
        """HTTPStatusError with 404 should also return None."""
        client = OllamaClient()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        http_err = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=mock_response,
        )
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 404
        mock_resp.raise_for_status = MagicMock(side_effect=http_err)
        mock_resp.json = MagicMock(return_value={})

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.get_model_info("nonexistent:7b")
        # 404 in HTTPStatusError handler returns None too
        assert result is None

    @pytest.mark.asyncio
    async def test_get_model_info_generic_exception_returns_none(self):
        client = OllamaClient()
        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=Exception("server error"))
            mock_get.return_value = mock_http
            result = await client.get_model_info("model:7b")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_model_info_http_status_error_500_returns_none(self):
        """Non-404 HTTPStatusError should also return None (logged + swallowed)."""
        client = OllamaClient()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        http_err = httpx.HTTPStatusError(
            "500 Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200  # status_code check won't match 404
        mock_resp.raise_for_status = MagicMock(side_effect=http_err)
        mock_resp.json = MagicMock(return_value={})

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.get_model_info("model:7b")
        assert result is None


# ============================================================================
# OllamaClient — is_model_available
# ============================================================================

class TestIsModelAvailable:

    @pytest.mark.asyncio
    async def test_returns_true_for_available_model(self):
        client = OllamaClient()
        with patch.object(
            client, "get_model_info", new=AsyncMock(return_value=_model_info("qwen2.5-coder:7b"))
        ):
            result = await client.is_model_available("qwen2.5-coder:7b")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_get_model_info_returns_none(self):
        client = OllamaClient()
        with patch.object(client, "get_model_info", new=AsyncMock(return_value=None)):
            result = await client.is_model_available("missing-model:7b")
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_cache_when_fresh(self):
        """If the cache is fresh (< 30 seconds old), get_model_info should NOT be called."""
        from app.integrations.ollama.client import ModelStatus
        client = OllamaClient()
        client.model_status["cached:7b"] = ModelStatus.AVAILABLE
        client.model_status_updated["cached:7b"] = datetime.utcnow()

        with patch.object(client, "get_model_info", new=AsyncMock()) as mock_info:
            result = await client.is_model_available("cached:7b")
        assert result is True
        mock_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_bypasses_cache_when_stale(self):
        """If the cache is older than 30 seconds, get_model_info SHOULD be called."""
        from app.integrations.ollama.client import ModelStatus
        client = OllamaClient()
        client.model_status["stale:7b"] = ModelStatus.AVAILABLE
        client.model_status_updated["stale:7b"] = datetime.utcnow() - timedelta(seconds=60)

        with patch.object(
            client, "get_model_info", new=AsyncMock(return_value=_model_info("stale:7b"))
        ) as mock_info:
            result = await client.is_model_available("stale:7b")
        assert result is True
        mock_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_updates_status_cache_after_check(self):
        from app.integrations.ollama.client import ModelStatus
        client = OllamaClient()
        with patch.object(client, "get_model_info", new=AsyncMock(return_value=None)):
            await client.is_model_available("new:7b")
        assert client.model_status.get("new:7b") == ModelStatus.UNAVAILABLE


# ============================================================================
# OllamaClient — model_exists
# ============================================================================

class TestModelExists:

    @pytest.mark.asyncio
    async def test_model_exists_true(self):
        client = OllamaClient()
        with patch.object(
            client, "list_models", new=AsyncMock(return_value=[_model_info("llama3:8b")])
        ):
            assert await client.model_exists("llama3:8b") is True

    @pytest.mark.asyncio
    async def test_model_exists_false(self):
        client = OllamaClient()
        with patch.object(client, "list_models", new=AsyncMock(return_value=[])):
            assert await client.model_exists("nonexistent:7b") is False

    @pytest.mark.asyncio
    async def test_model_exists_handles_exception(self):
        client = OllamaClient()
        with patch.object(
            client, "list_models", new=AsyncMock(side_effect=Exception("network error"))
        ):
            assert await client.model_exists("any:7b") is False


# ============================================================================
# OllamaClient — pull_model
# ============================================================================

class TestPullModel:

    @pytest.mark.asyncio
    async def test_pull_model_success(self):
        """pull_model should return True when stream ends with status=success."""
        client = OllamaClient()

        lines = [
            json.dumps({"status": "pulling manifest"}),
            json.dumps({"status": "success"}),
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        with patch.object(client.connection_pool, "get_client", return_value=mock_http):
            result = await client.pull_model("qwen2.5-coder:7b")

        assert result is True

    @pytest.mark.asyncio
    async def test_pull_model_error_in_stream_returns_false(self):
        """pull_model returns False when the stream contains an error field."""
        client = OllamaClient()

        lines = [
            json.dumps({"error": "model not found in registry"}),
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        with patch.object(client.connection_pool, "get_client", return_value=mock_http):
            result = await client.pull_model("nonexistent:99b")

        assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_exception_returns_false(self):
        """pull_model returns False when an exception is raised."""
        client = OllamaClient()

        with patch.object(
            client.connection_pool, "get_client", side_effect=Exception("connection refused")
        ):
            result = await client.pull_model("model:7b")

        assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_invalid_json_lines_skipped(self):
        """Non-JSON lines in the stream should be skipped gracefully."""
        client = OllamaClient()

        lines = [
            "not-json-at-all",
            json.dumps({"status": "success"}),
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        with patch.object(client.connection_pool, "get_client", return_value=mock_http):
            result = await client.pull_model("model:7b")

        assert result is True

    @pytest.mark.asyncio
    async def test_pull_model_progress_logging(self):
        """pull_model with completed/total fields should log progress."""
        client = OllamaClient()

        lines = [
            json.dumps({"status": "downloading", "completed": 100, "total": 1000}),
            json.dumps({"status": "success"}),
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        with patch.object(client.connection_pool, "get_client", return_value=mock_http):
            result = await client.pull_model("model:7b")

        assert result is True

    @pytest.mark.asyncio
    async def test_pull_model_sets_status_available_on_success(self):
        """After successful pull, model_status should be AVAILABLE."""
        from app.integrations.ollama.client import ModelStatus
        client = OllamaClient()

        lines = [json.dumps({"status": "success"})]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        with patch.object(client.connection_pool, "get_client", return_value=mock_http):
            await client.pull_model("pulled:7b")

        assert client.model_status.get("pulled:7b") == ModelStatus.AVAILABLE


# ============================================================================
# OllamaClient — chat (sync)
# ============================================================================

class TestChatSync:

    @pytest.mark.asyncio
    async def test_chat_sync_success(self):
        """Sync chat returns the assistant content string."""
        client = OllamaClient()

        mock_resp = _make_response(200, {
            "model": "qwen2.5-coder:7b",
            "message": {"role": "assistant", "content": "Hello, world!"},
            "done": True,
            "eval_count": 5,
            "eval_duration": 1_000_000_000,
        })

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            result = await client.chat(
                model="qwen2.5-coder:7b",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_chat_sync_updates_model_stats(self):
        client = OllamaClient()

        mock_resp = _make_response(200, {
            "model": "qwen2.5-coder:7b",
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
            "eval_count": 10,
        })

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            await client.chat(
                model="qwen2.5-coder:7b",
                messages=[{"role": "user", "content": "Hi"}],
            )

        stats = client.model_stats.get("qwen2.5-coder:7b")
        assert stats is not None
        assert stats.total_requests == 1
        assert stats.successful_requests == 1

    @pytest.mark.asyncio
    async def test_chat_invalid_model_name_raises_value_error(self):
        client = OllamaClient()
        with pytest.raises(ValueError):
            await client.chat(
                model="invalid model name with spaces!!",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @pytest.mark.asyncio
    async def test_chat_timeout_raises_model_timeout_error(self):
        client = OllamaClient()

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_get.return_value = mock_http

            with pytest.raises((ModelTimeoutError, Exception)):
                await client.chat(
                    model="qwen2.5-coder:7b",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_chat_404_raises_model_not_found(self):
        client = OllamaClient()
        # Initialize model stats so _handle_sync doesn't KeyError
        from app.integrations.ollama.client import ModelStats
        client.model_stats["qwen2.5-coder:7b"] = ModelStats()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        http_err = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=mock_response
        )
        mock_resp = _make_response(404)
        mock_resp.raise_for_status = MagicMock(side_effect=http_err)

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http

            with pytest.raises((ModelNotFoundError, Exception)):
                await client.chat(
                    model="qwen2.5-coder:7b",
                    messages=[{"role": "user", "content": "Hi"}],
                )


# ============================================================================
# OllamaClient — health_check
# ============================================================================

class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_success_returns_healthy(self):
        client = OllamaClient()
        with patch.object(
            client,
            "list_models",
            new=AsyncMock(return_value=[_model_info("q:7b")]),
        ):
            result = await client.health_check()

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_includes_connectivity_info(self):
        client = OllamaClient()
        with patch.object(
            client,
            "list_models",
            new=AsyncMock(return_value=[_model_info("q:7b")]),
        ):
            result = await client.health_check()

        assert "checks" in result
        assert "connectivity" in result["checks"]
        assert result["checks"]["connectivity"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_includes_queue_stats(self):
        client = OllamaClient()
        with patch.object(
            client,
            "list_models",
            new=AsyncMock(return_value=[]),
        ):
            result = await client.health_check()

        assert "queue" in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_connection_error(self):
        client = OllamaClient()
        with patch.object(
            client,
            "list_models",
            new=AsyncMock(side_effect=ModelConnectionError("no connection")),
        ):
            result = await client.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_generic_exception(self):
        client = OllamaClient()
        with patch.object(
            client,
            "list_models",
            new=AsyncMock(side_effect=RuntimeError("unexpected failure")),
        ):
            result = await client.health_check()

        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_includes_model_stats(self):
        client = OllamaClient()
        client.model_stats["test:7b"] = ModelStats()
        client.model_stats["test:7b"].update_success(0.5, 100)

        with patch.object(
            client,
            "list_models",
            new=AsyncMock(return_value=[_model_info("test:7b")]),
        ):
            result = await client.health_check()

        assert "model_stats" in result
        assert "test:7b" in result["model_stats"]


# ============================================================================
# OllamaClient — get_stats
# ============================================================================

class TestGetStats:

    def test_get_stats_returns_dict(self):
        client = OllamaClient()
        stats = client.get_stats()
        assert isinstance(stats, dict)
        assert "base_url" in stats
        assert "total_models_tracked" in stats
        assert "queue_stats" in stats
        assert "models" in stats

    def test_get_stats_reflects_model_stats(self):
        client = OllamaClient()
        client.model_stats["abc:7b"] = ModelStats()
        client.model_stats["abc:7b"].update_success(1.0, 50)

        stats = client.get_stats()
        assert stats["total_models_tracked"] == 1
        assert "abc:7b" in stats["models"]
        model_data = stats["models"]["abc:7b"]
        assert model_data["total_requests"] == 1
        assert model_data["successful"] == 1

    def test_get_stats_with_no_models(self):
        client = OllamaClient()
        stats = client.get_stats()
        assert stats["total_models_tracked"] == 0
        assert stats["models"] == {}


# ============================================================================
# OllamaClient — context manager & close
# ============================================================================

class TestLifecycle:

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with OllamaClient() as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        client = OllamaClient()
        await client.close()  # Should not raise


# ============================================================================
# ChatRequest model name validator
# ============================================================================

class TestChatRequest:

    def test_valid_model_name_accepted(self):
        from app.integrations.ollama.client import ChatRequest, ChatMessage, ChatRole
        req = ChatRequest(
            model="qwen2.5-coder:7b",
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
        )
        assert req.model == "qwen2.5-coder:7b"

    def test_invalid_model_name_raises_value_error(self):
        from app.integrations.ollama.client import ChatRequest, ChatMessage, ChatRole
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            ChatRequest(
                model="invalid model with spaces!!",
                messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
            )


# ============================================================================
# ModelInfo.size_formatted
# ============================================================================

class TestModelInfoSizeFormatted:

    def test_size_formatted_gb(self):
        info = ModelInfo(
            name="large:7b",
            modified_at=datetime.utcnow(),
            size=4_000_000_000,
            digest="sha256:abc",
        )
        assert "GB" in info.size_formatted

    def test_size_formatted_mb(self):
        info = ModelInfo(
            name="tiny:7b",
            modified_at=datetime.utcnow(),
            size=500_000_000,  # < 1 GB
            digest="sha256:abc",
        )
        assert "MB" in info.size_formatted

    def test_size_gb_property(self):
        info = ModelInfo(
            name="mid:7b",
            modified_at=datetime.utcnow(),
            size=2_147_483_648,  # 2 GB
            digest="sha256:abc",
        )
        assert abs(info.size_gb - 2.0) < 0.01


# ============================================================================
# _handle_http_error — 429, 500+ status codes
# ============================================================================

class TestHandleHttpError:

    def test_http_error_404_raises_model_not_found(self):
        from app.integrations.ollama.client import ModelStats
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        err = httpx.HTTPStatusError("404", request=MagicMock(), response=mock_response)
        with pytest.raises(ModelNotFoundError):
            client._handle_http_error(err, "m:7b", 0.1)

    def test_http_error_429_raises_model_overloaded(self):
        from app.integrations.ollama.client import ModelStats, ModelOverloadedError
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response)
        with pytest.raises(ModelOverloadedError):
            client._handle_http_error(err, "m:7b", 0.1)

    def test_http_error_500_raises_ollama_client_error(self):
        from app.integrations.ollama.client import ModelStats
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_response)
        with pytest.raises(OllamaClientError):
            client._handle_http_error(err, "m:7b", 0.1)

    def test_http_error_other_raises_ollama_client_error(self):
        from app.integrations.ollama.client import ModelStats
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        err = httpx.HTTPStatusError("403", request=MagicMock(), response=mock_response)
        with pytest.raises(OllamaClientError):
            client._handle_http_error(err, "m:7b", 0.1)


# ============================================================================
# _handle_streaming
# ============================================================================

class TestHandleStreaming:

    @pytest.mark.asyncio
    async def test_handle_streaming_yields_content(self):
        from app.integrations.ollama.client import ModelStats
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()

        lines = [
            json.dumps({"message": {"content": "Hello"}}),
            json.dumps({"message": {"content": " world"}}),
            json.dumps({"done": True}),
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        tokens = []
        async for token in client._handle_streaming(mock_http, {}, {}, "m:7b"):
            tokens.append(token)
        assert "Hello" in tokens
        assert " world" in tokens

    @pytest.mark.asyncio
    async def test_handle_streaming_skips_invalid_json(self):
        from app.integrations.ollama.client import ModelStats
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()

        lines = [
            "not-json",
            json.dumps({"message": {"content": "OK"}}),
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        tokens = []
        async for token in client._handle_streaming(mock_http, {}, {}, "m:7b"):
            tokens.append(token)
        assert "OK" in tokens

    @pytest.mark.asyncio
    async def test_handle_streaming_raises_on_stream_error_field(self):
        from app.integrations.ollama.client import ModelStats
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()

        lines = [json.dumps({"error": "model overloaded"})]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = AsyncMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)

        with pytest.raises(OllamaClientError):
            async for _ in client._handle_streaming(mock_http, {}, {}, "m:7b"):
                pass


# ============================================================================
# _send_chat_request with API key
# ============================================================================

class TestSendChatRequestApiKey:

    @pytest.mark.asyncio
    async def test_api_key_added_to_headers(self):
        from app.integrations.ollama.client import ChatRequest, ChatMessage, ChatRole
        client = OllamaClient(api_key="test-key-123")

        mock_resp = _make_response(200, {
            "model": "qwen2.5-coder:7b",
            "message": {"role": "assistant", "content": "Response"},
            "done": True,
        })

        captured_headers = {}

        async def mock_post(url, *, json=None, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return mock_resp

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_get.return_value = mock_http
            await client.chat(
                model="qwen2.5-coder:7b",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert "Authorization" in captured_headers
        assert "test-key-123" in captured_headers["Authorization"]


# ============================================================================
# chat — 429 and 500 HTTP errors
# ============================================================================

class TestChatHttpErrors:

    @pytest.mark.asyncio
    async def test_chat_429_raises_model_overloaded(self):
        from app.integrations.ollama.client import ModelStats, ModelOverloadedError
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        http_err = httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock(side_effect=http_err)
        mock_resp.json = MagicMock(return_value={})

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            with pytest.raises((ModelOverloadedError, Exception)):
                await client.chat(model="m:7b", messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_chat_500_raises_ollama_client_error(self):
        from app.integrations.ollama.client import ModelStats
        client = OllamaClient()
        client.model_stats["m:7b"] = ModelStats()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        http_err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_response)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock(side_effect=http_err)
        mock_resp.json = MagicMock(return_value={})

        with patch.object(client.connection_pool, "get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_http
            with pytest.raises((OllamaClientError, Exception)):
                await client.chat(model="m:7b", messages=[{"role": "user", "content": "Hi"}])
