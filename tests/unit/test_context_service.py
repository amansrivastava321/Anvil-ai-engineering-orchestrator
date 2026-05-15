"""Tests for app.services.context_service — ContextAssembler and data classes."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from app.services.context_service import (
    ContextChunk,
    ContextBudget,
    ContextSource,
    ContextPriority,
    ContextMode,
    AssembledContext,
    ContextAssembler,
)


# ── ContextChunk ──────────────────────────────────────────────────────────────

def test_context_chunk_estimates_tokens():
    chunk = ContextChunk(
        content="x" * 400,
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
    )
    assert chunk.token_estimate == 100  # 400 / 4


def test_context_chunk_explicit_token_estimate():
    chunk = ContextChunk(
        content="hello",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
        token_estimate=50,
    )
    assert chunk.token_estimate == 50


def test_context_chunk_estimate_tokens_static():
    estimate = ContextChunk._estimate_tokens("a" * 800)
    assert estimate == 200


# ── ContextBudget ─────────────────────────────────────────────────────────────

def test_context_budget_allocated_sums_categories():
    budget = ContextBudget(
        total_limit=8000,
        system_instructions=500,
        user_prompt=200,
        code_files=1000,
    )
    assert budget.allocated == 1700


def test_context_budget_available_subtracts_allocated_and_safety():
    budget = ContextBudget(
        total_limit=8000,
        system_instructions=1000,
        reserved_safety=200,
    )
    assert budget.available == 6800


def test_context_budget_available_never_goes_negative():
    budget = ContextBudget(total_limit=100, code_files=200, reserved_safety=50)
    assert budget.available == 0


def test_context_budget_allocate_succeeds_when_space_available():
    budget = ContextBudget(total_limit=8000)
    success = budget.allocate("code_files", 500)
    assert success is True
    assert budget.code_files == 500


def test_context_budget_allocate_fails_when_no_space():
    budget = ContextBudget(total_limit=500, code_files=400, reserved_safety=50)
    success = budget.allocate("documentation", 200)
    assert success is False


def test_context_budget_allocate_unknown_category_returns_false():
    budget = ContextBudget(total_limit=8000)
    assert budget.allocate("nonexistent_category", 100) is False


# ── ContextAssembler ──────────────────────────────────────────────────────────

@pytest.fixture
def assembler(tmp_path):
    with patch("app.services.context_service.get_default_parser") as mock_parser, \
         patch("app.services.context_service.get_skillfile_client") as mock_skill:
        mock_parser.return_value = MagicMock()
        mock_skill.return_value = MagicMock()
        assembler = ContextAssembler()
        assembler._tmp_path = str(tmp_path)
        return assembler


def test_assembler_initialization():
    with patch("app.services.context_service.get_default_parser") as mp, \
         patch("app.services.context_service.get_skillfile_client") as ms:
        mp.return_value = MagicMock()
        ms.return_value = MagicMock()
        a = ContextAssembler()
        assert a is not None


def test_assembler_enable_skill_injection():
    with patch("app.services.context_service.get_default_parser") as mp, \
         patch("app.services.context_service.get_skillfile_client") as ms:
        mp.return_value = MagicMock()
        ms.return_value = MagicMock()
        a = ContextAssembler()
        a.enable_skill_injection(True)
        a.enable_skill_injection(False)  # should not raise


def test_assembler_set_skill_limits():
    with patch("app.services.context_service.get_default_parser") as mp, \
         patch("app.services.context_service.get_skillfile_client") as ms:
        mp.return_value = MagicMock()
        ms.return_value = MagicMock()
        a = ContextAssembler()
        a.set_skill_limits(max_skills=3, max_tokens_per_skill=500)


@pytest.mark.asyncio
async def test_assemble_context_returns_assembled_context(tmp_path):
    (tmp_path / "main.py").write_text("def hello(): pass\n")
    from app.utils.validators import PathValidator
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        with patch("app.services.context_service.get_default_parser") as mp, \
             patch("app.services.context_service.get_skillfile_client") as ms:
            mp.return_value = MagicMock()
            ms.return_value = MagicMock()
            assembler = ContextAssembler()
            with patch.object(assembler, "_load_graphify_context", new=AsyncMock(return_value=[])):
                with patch.object(assembler, "_load_skill_context", new=AsyncMock(return_value=([], []))):
                    result = await assembler.assemble_context(
                        user_prompt="Fix the bug in hello()",
                        repo_path=str(tmp_path),
                        task_type="debugging",
                    )
        assert isinstance(result, AssembledContext)
        assert isinstance(result.system_prompt, str)
        assert isinstance(result.user_prompt, str)
        assert result.total_tokens >= 0
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


@pytest.mark.asyncio
async def test_assemble_context_with_files_included(tmp_path):
    (tmp_path / "module.py").write_text("class Foo:\n    pass\n")
    from app.utils.validators import PathValidator
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        with patch("app.services.context_service.get_default_parser") as mp, \
             patch("app.services.context_service.get_skillfile_client") as ms:
            mp.return_value = MagicMock()
            ms.return_value = MagicMock()
            assembler = ContextAssembler()
            with patch.object(assembler, "_load_graphify_context", new=AsyncMock(return_value=[])):
                with patch.object(assembler, "_load_skill_context", new=AsyncMock(return_value=([], []))):
                    result = await assembler.assemble_context(
                        user_prompt="Review this code",
                        repo_path=str(tmp_path),
                        task_type="code_review",
                    )
        assert isinstance(result.files_included, list)
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


@pytest.mark.asyncio
async def test_estimate_tokens_returns_dict(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    with patch("app.services.context_service.get_default_parser") as mp, \
         patch("app.services.context_service.get_skillfile_client") as ms:
        mp.return_value = MagicMock()
        ms.return_value = MagicMock()
        assembler = ContextAssembler()
        result = await assembler.estimate_tokens(
            repo_path=str(tmp_path),
            prompt="Analyze this",
        )
    assert isinstance(result, dict)
    assert len(result) > 0


# ── ContextSource / ContextPriority / ContextMode enums ───────────────────────

def test_context_source_values():
    assert ContextSource.CODE_FILE.value == "code_file"
    assert ContextSource.DOCUMENTATION.value == "documentation"


def test_context_priority_ordering_values():
    assert ContextPriority.HIGH.value == "high"
    assert ContextPriority.LOW.value == "low"


def test_context_mode_values():
    assert ContextMode.BALANCED.value == "balanced"
    assert ContextMode.CODE_ONLY.value == "code_only"
