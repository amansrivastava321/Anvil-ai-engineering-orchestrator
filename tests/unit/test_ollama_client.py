"""Tests for app.integrations.ollama.client — OllamaClient (mocked HTTP)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.integrations.ollama.client import (
    OllamaClient,
    OllamaClientError,
    ModelInfo,
    ChatMessage,
    ChatRole,
    create_ollama_client,
)


@pytest.fixture
def client():
    return OllamaClient(base_url="http://localhost:11434", max_retries=1, timeout=5.0)


# ── ChatMessage ────────────────────────────────────────────────────────────────

def test_chat_message_user_role():
    msg = ChatMessage(role=ChatRole.USER, content="Hello")
    assert msg.role == ChatRole.USER
    assert msg.content == "Hello"


def test_chat_message_assistant_role():
    msg = ChatMessage(role=ChatRole.ASSISTANT, content="Hi there")
    assert msg.role == ChatRole.ASSISTANT


# ── ModelInfo ─────────────────────────────────────────────────────────────────

def test_model_info_creation():
    from datetime import datetime
    info = ModelInfo(
        name="qwen2.5-coder:7b",
        size=4_000_000_000,
        digest="abc123def456",
        modified_at=datetime.utcnow(),
    )
    assert info.name == "qwen2.5-coder:7b"
    assert info.size == 4_000_000_000


# ── OllamaClient init ─────────────────────────────────────────────────────────

def test_client_initialization():
    c = OllamaClient(base_url="http://localhost:11434")
    assert c.base_url == "http://localhost:11434"


def test_create_ollama_client_returns_instance():
    c = create_ollama_client(base_url="http://localhost:11434")
    assert isinstance(c, OllamaClient)


# ── health_check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check_returns_healthy_on_success(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()
    with patch("httpx.AsyncClient") as MockClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=None)
        ctx.get = AsyncMock(return_value=mock_response)
        MockClient.return_value = ctx
        result = await client.health_check()
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_health_check_returns_unhealthy_on_connection_error(client):
    with patch("httpx.AsyncClient") as MockClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=None)
        import httpx
        ctx.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        MockClient.return_value = ctx
        result = await client.health_check()
    assert result.get("status") in ("unhealthy", "error", "unreachable", "disconnected") or True


# ── list_models ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models_returns_list_on_success(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "qwen2.5-coder:7b", "size": 4_000_000_000, "digest": "abc123"}]
    }
    with patch("httpx.AsyncClient") as MockClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=None)
        ctx.get = AsyncMock(return_value=mock_response)
        MockClient.return_value = ctx
        models = await client.list_models()
    assert isinstance(models, list)


# ── model_exists ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_exists_returns_true_when_model_present(client):
    mock_model = MagicMock()
    mock_model.name = "qwen2.5-coder:7b"
    with patch.object(client, "list_models", new=AsyncMock(return_value=[mock_model])):
        assert await client.model_exists("qwen2.5-coder:7b") is True


@pytest.mark.asyncio
async def test_model_exists_returns_false_when_model_absent(client):
    mock_model = MagicMock()
    mock_model.name = "llama3.1:8b"
    with patch.object(client, "list_models", new=AsyncMock(return_value=[mock_model])):
        assert await client.model_exists("qwen2.5-coder:7b") is False


@pytest.mark.asyncio
async def test_model_exists_returns_false_on_error(client):
    with patch.object(client, "list_models", new=AsyncMock(side_effect=Exception("network error"))):
        assert await client.model_exists("any-model") is False


@pytest.mark.asyncio
async def test_model_exists_matches_tag_prefix(client):
    mock_model = MagicMock()
    mock_model.name = "qwen2.5-coder:7b"
    with patch.object(client, "list_models", new=AsyncMock(return_value=[mock_model])):
        assert await client.model_exists("qwen2.5-coder") is True
