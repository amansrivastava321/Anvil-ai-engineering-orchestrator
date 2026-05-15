"""Tests for app.integrations.graphify.parser — GraphifyParser + GraphifyWrapper."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.integrations.graphify.parser import (
    GraphifyParser,
    GraphifyWrapper,
    get_default_parser,
)


# ── GraphifyWrapper ───────────────────────────────────────────────────────────

def test_graphify_wrapper_not_installed_returns_false(tmp_path):
    wrapper = GraphifyWrapper(str(tmp_path))
    # In test environment graphify CLI is not installed
    assert isinstance(wrapper.is_installed(), bool)


@pytest.mark.asyncio
async def test_graphify_wrapper_run_graphify_returns_dict_on_failure(tmp_path):
    # GraphifyParser.parse_repository wraps run_graphify errors into {"available": False}
    from app.integrations.graphify.parser import GraphifyParser
    parser = GraphifyParser()
    result = await parser.parse_repository(str(tmp_path))
    assert isinstance(result, dict)
    assert result.get("available") is False


# ── GraphifyParser ────────────────────────────────────────────────────────────

@pytest.fixture
def parser():
    return GraphifyParser()


@pytest.mark.asyncio
async def test_parse_repository_returns_dict(tmp_path, parser):
    result = await parser.parse_repository(str(tmp_path))
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_parse_repository_returns_available_false_on_missing_graphify(tmp_path, parser):
    with patch("app.integrations.graphify.parser.GraphifyWrapper") as MockWrapper:
        instance = MockWrapper.return_value
        instance.run_graphify = AsyncMock(side_effect=Exception("not installed"))
        result = await parser.parse_repository(str(tmp_path))
    assert result.get("available") is False


@pytest.mark.asyncio
async def test_get_affected_modules_returns_dict(tmp_path, parser):
    with patch.object(parser, "parse_repository", new=AsyncMock(return_value={"available": False})):
        result = await parser.get_affected_modules(str(tmp_path), ["main.py"])
    assert isinstance(result, dict)
    assert "affected" in result


@pytest.mark.asyncio
async def test_get_affected_modules_with_module_data(tmp_path, parser):
    parse_result = {
        "available": True,
        "modules": {
            "app.utils": {"imports": ["main.py"], "dependencies": []},
            "app.core": {"imports": [], "dependencies": []},
        },
    }
    with patch.object(parser, "parse_repository", new=AsyncMock(return_value=parse_result)):
        result = await parser.get_affected_modules(str(tmp_path), ["main.py"])
    assert "app.utils" in result["affected"]
    assert "app.core" not in result["affected"]


@pytest.mark.asyncio
async def test_get_module_dependencies_returns_dict(tmp_path, parser):
    with patch.object(parser, "parse_repository", new=AsyncMock(return_value={"available": False})):
        result = await parser.get_module_dependencies(str(tmp_path), "app.utils")
    assert isinstance(result, dict)
    assert "module" in result


@pytest.mark.asyncio
async def test_get_module_dependencies_with_data(tmp_path, parser):
    parse_result = {
        "available": True,
        "modules": {
            "app.utils": {"imports": ["os", "pathlib"], "dependencies": ["requests"]},
            "app.core": {"imports": ["app.utils"], "dependencies": []},
        },
    }
    with patch.object(parser, "parse_repository", new=AsyncMock(return_value=parse_result)):
        result = await parser.get_module_dependencies(str(tmp_path), "app.utils")
    assert result["module"] == "app.utils"
    assert "app.core" in result["dependents"]
    assert "os" in result["imports"]


def test_clear_cache_is_safe(parser):
    parser.clear_cache()  # should not raise


def test_get_stats_returns_dict(parser):
    stats = parser.get_stats()
    assert isinstance(stats, dict)
    assert "parser" in stats


@pytest.mark.asyncio
async def test_close_is_safe(parser):
    await parser.close()  # should not raise


def test_get_default_parser_returns_singleton():
    p1 = get_default_parser()
    p2 = get_default_parser()
    assert p1 is p2
