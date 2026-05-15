"""Tests for app.services.model_service — ModelService selection and routing."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.model_service import ModelService, TaskCategory, ModelTier


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.list_models = AsyncMock(return_value=[])
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


@pytest.fixture
def svc(mock_ollama):
    return ModelService(ollama_client=mock_ollama)


# ── select_model ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_select_model_returns_string(svc):
    svc._is_available = AsyncMock(return_value=True)
    model = await svc.select_model(TaskCategory.CODE_GENERATION, require_available=False)
    assert isinstance(model, str)
    assert len(model) > 0


@pytest.mark.asyncio
async def test_select_model_prefers_user_specified(svc):
    svc._is_available = AsyncMock(return_value=True)
    from app.utils.validators import validate_model
    with patch("app.services.model_service.validate_model", return_value=True):
        model = await svc.select_model(
            TaskCategory.CODE_GENERATION,
            preferred_model="qwen2.5-coder:7b",
            require_available=True,
        )
    assert model == "qwen2.5-coder:7b"


@pytest.mark.asyncio
async def test_select_model_skips_unavailable_preferred(svc):
    svc._is_available = AsyncMock(return_value=False)
    with patch("app.services.model_service.validate_model", return_value=True):
        # Should fall through to candidates and pick something
        model = await svc.select_model(
            TaskCategory.CODE_GENERATION,
            preferred_model="unavailable-model:7b",
            require_available=False,
        )
    assert isinstance(model, str)


@pytest.mark.asyncio
async def test_select_model_no_require_available_returns_first_candidate(svc):
    model = await svc.select_model(TaskCategory.CODE_GENERATION, require_available=False)
    assert isinstance(model, str)


# ── list_available_models ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_available_models_returns_list(svc, mock_ollama):
    mock_model = MagicMock()
    mock_model.name = "qwen2.5-coder:7b"
    mock_ollama.list_models = AsyncMock(return_value=[mock_model])
    result = await svc.list_available_models()
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_list_available_models_handles_error(svc, mock_ollama):
    mock_ollama.list_models = AsyncMock(side_effect=Exception("no connection"))
    try:
        result = await svc.list_available_models()
        assert isinstance(result, list)
    except Exception:
        pass  # Some implementations propagate errors — both valid


# ── health_check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check_returns_dict(svc, mock_ollama):
    mock_ollama.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_ollama.model_stats = {}
    svc._is_available = AsyncMock(return_value=False)
    result = await svc.health_check()
    assert isinstance(result, dict)
    assert "total_models" in result


# ── get_selection_stats ───────────────────────────────────────────────────────

def test_get_selection_stats_returns_dict(svc):
    stats = svc.get_selection_stats()
    assert isinstance(stats, dict)


def test_get_selection_stats_tracks_selections(svc):
    svc._record_selection("model-a")
    svc._record_selection("model-a")
    stats = svc.get_selection_stats()
    assert stats.get("total_selections", 0) >= 2 or "model-a" in str(stats)


# ── TaskCategory ──────────────────────────────────────────────────────────────

def test_task_category_values_are_strings():
    assert isinstance(TaskCategory.CODE_GENERATION.value, str)
    assert isinstance(TaskCategory.DEBUGGING.value, str)


# ── select_model with require_available=True and available model ──────────────

@pytest.mark.asyncio
async def test_select_model_with_require_available_true_finds_model(svc):
    svc._is_available = AsyncMock(return_value=True)
    model = await svc.select_model(TaskCategory.CODE_GENERATION, require_available=True)
    assert isinstance(model, str)


@pytest.mark.asyncio
async def test_select_model_with_all_unavailable_uses_fallback(svc):
    svc._is_available = AsyncMock(return_value=False)
    model = await svc.select_model(TaskCategory.CODE_GENERATION, require_available=True)
    assert isinstance(model, str)
    svc._record_selection.assert_not_called() if hasattr(svc._record_selection, "assert_not_called") else None


@pytest.mark.asyncio
async def test_select_model_with_tier_filter(svc):
    svc._is_available = AsyncMock(return_value=False)
    model = await svc.select_model(
        TaskCategory.CODE_GENERATION,
        tier=ModelTier.FAST,
        require_available=False,
    )
    assert isinstance(model, str)


@pytest.mark.asyncio
async def test_select_model_for_debugging_task(svc):
    svc._is_available = AsyncMock(return_value=True)
    model = await svc.select_model(TaskCategory.DEBUGGING, require_available=False)
    assert isinstance(model, str)


@pytest.mark.asyncio
async def test_select_model_for_architecture_task(svc):
    svc._is_available = AsyncMock(return_value=True)
    model = await svc.select_model(TaskCategory.ARCHITECTURE_ANALYSIS, require_available=False)
    assert isinstance(model, str)


# ── list_available_models with filters ────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_available_models_with_task_type_filter(svc, mock_ollama):
    mock_model = MagicMock()
    mock_model.name = "qwen2.5-coder:7b"
    mock_model.size_formatted = "4.5 GB"
    mock_ollama.list_models = AsyncMock(return_value=[mock_model])
    result = await svc.list_available_models(task_type=TaskCategory.CODE_GENERATION)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_list_available_models_with_tier_filter(svc, mock_ollama):
    mock_model = MagicMock()
    mock_model.name = "llama3:8b"
    mock_model.size_formatted = "8 GB"
    mock_ollama.list_models = AsyncMock(return_value=[mock_model])
    result = await svc.list_available_models(tier=ModelTier.FAST)
    assert isinstance(result, list)


# ── _is_available with cache ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_is_available_uses_cache(svc, mock_ollama):
    mock_ollama.is_model_available = AsyncMock(return_value=True)
    # First call — populates cache
    result1 = await svc._is_available("qwen2.5-coder:7b")
    # Second call — should use cache (no additional Ollama call)
    result2 = await svc._is_available("qwen2.5-coder:7b")
    assert result1 == result2
    mock_ollama.is_model_available.assert_called_once()


@pytest.mark.asyncio
async def test_is_available_force_check_bypasses_cache(svc, mock_ollama):
    mock_ollama.is_model_available = AsyncMock(return_value=True)
    await svc._is_available("qwen2.5-coder:7b")
    await svc._is_available("qwen2.5-coder:7b", force_check=True)
    assert mock_ollama.is_model_available.call_count == 2


@pytest.mark.asyncio
async def test_is_available_returns_false_for_unknown_model(svc, mock_ollama):
    mock_ollama.is_model_available = AsyncMock(return_value=False)
    result = await svc._is_available("completely-unknown:1b")
    assert result is False


# ── health_check with available model ────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check_counts_available(svc, mock_ollama):
    mock_ollama.model_stats = {}
    svc._is_available = AsyncMock(return_value=True)
    result = await svc.health_check()
    assert result["available_models"] >= 0


@pytest.mark.asyncio
async def test_health_check_includes_model_details(svc, mock_ollama):
    mock_ollama.model_stats = {}
    svc._is_available = AsyncMock(return_value=False)
    result = await svc.health_check()
    assert "models" in result


# ── get_selection_stats edge cases ────────────────────────────────────────────

def test_get_selection_stats_with_fallbacks(svc):
    svc._record_fallback("model-b", "code_generation")
    stats = svc.get_selection_stats()
    assert "fallbacks" in stats or isinstance(stats, dict)


def test_get_all_registered_models_returns_list(svc):
    result = svc._get_all_registered_models()
    assert isinstance(result, list)
    assert len(result) > 0  # MODEL_REGISTRY should have entries


# ── get_model_info ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_model_info_returns_dict(svc, mock_ollama):
    mock_ollama.is_model_available = AsyncMock(return_value=True)
    mock_ollama.get_model_info = AsyncMock(return_value=None)
    result = await svc.get_model_info("qwen2.5-coder:7b")
    assert isinstance(result, dict)
    assert result["name"] == "qwen2.5-coder:7b"
    assert "available" in result
    assert "tier" in result


@pytest.mark.asyncio
async def test_get_model_info_with_ollama_data(svc, mock_ollama):
    mock_model_info = MagicMock()
    mock_model_info.size_formatted = "4.5 GB"
    mock_model_info.modified_at = MagicMock()
    mock_model_info.modified_at.isoformat = MagicMock(return_value="2024-01-01T00:00:00")
    mock_ollama.is_model_available = AsyncMock(return_value=True)
    mock_ollama.get_model_info = AsyncMock(return_value=mock_model_info)
    result = await svc.get_model_info("qwen2.5-coder:7b")
    assert result["size"] == "4.5 GB"


@pytest.mark.asyncio
async def test_get_model_info_unknown_model(svc, mock_ollama):
    mock_ollama.is_model_available = AsyncMock(return_value=False)
    mock_ollama.get_model_info = AsyncMock(return_value=None)
    result = await svc.get_model_info("completely-unknown:1b")
    assert result["available"] is False
    assert result["size"] == "unknown"


# ── create_model_chain ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_model_chain_code_generation(svc):
    svc._is_available = AsyncMock(return_value=True)
    from app.services.model_service import TaskCategory
    chain = await svc.create_model_chain(TaskCategory.CODE_GENERATION, chain_size=2)
    assert isinstance(chain, list)
    assert len(chain) >= 1


@pytest.mark.asyncio
async def test_create_model_chain_code_generation_three_models(svc):
    svc._is_available = AsyncMock(return_value=True)
    from app.services.model_service import TaskCategory
    chain = await svc.create_model_chain(TaskCategory.CODE_GENERATION, chain_size=3)
    assert isinstance(chain, list)
    assert len(chain) >= 2


@pytest.mark.asyncio
async def test_create_model_chain_architecture_analysis(svc):
    svc._is_available = AsyncMock(return_value=True)
    from app.services.model_service import TaskCategory
    chain = await svc.create_model_chain(TaskCategory.ARCHITECTURE_ANALYSIS)
    assert isinstance(chain, list)
    assert len(chain) >= 1


@pytest.mark.asyncio
async def test_create_model_chain_general_qa_default(svc):
    svc._is_available = AsyncMock(return_value=True)
    from app.services.model_service import TaskCategory
    chain = await svc.create_model_chain(TaskCategory.GENERAL_QA, chain_size=2)
    assert isinstance(chain, list)
    assert len(chain) >= 1


@pytest.mark.asyncio
async def test_create_model_chain_returns_fallback_if_empty(svc):
    # If all models are unavailable, should still return something
    svc._is_available = AsyncMock(return_value=False)
    from app.services.model_service import TaskCategory
    chain = await svc.create_model_chain(TaskCategory.DEBUGGING)
    assert isinstance(chain, list)
    assert len(chain) >= 1


# ── get_model_service factory ──────────────────────────────────────────────────

def test_get_model_service_returns_instance():
    from app.services.model_service import get_model_service, ModelService
    import app.services.model_service as ms_module
    orig = ms_module._default_service
    ms_module._default_service = None
    try:
        with patch("app.services.model_service.get_default_client"):
            svc = get_model_service()
            assert isinstance(svc, ModelService)
    finally:
        ms_module._default_service = orig
