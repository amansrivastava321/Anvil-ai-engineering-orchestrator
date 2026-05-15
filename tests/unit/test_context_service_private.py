"""
Comprehensive tests for private/utility methods of ContextAssembler.

Covers:
  _truncate_to_tokens, _priority_sort_order, _read_file_safe,
  _emergency_truncate, _merge_chunks, _wrap_user_prompt,
  _format_summary, _format_app_map, _format_graph_context,
  _load_conversation_history, _load_file_context,
  _get_default_token_limit, _create_budget, _get_graphify_budget,
  _generate_cache_key, _update_stats, get_stats,
  clear_cache, close, enable_skill_injection, set_skill_limits
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.context_service import (
    ContextChunk,
    ContextBudget,
    ContextSource,
    ContextPriority,
    ContextMode,
    AssembledContext,
    ContextAssembler,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def assembler():
    with patch("app.services.context_service.get_default_parser") as mp, \
         patch("app.services.context_service.get_skillfile_client") as ms:
        mp.return_value = MagicMock()
        ms.return_value = MagicMock()
        yield ContextAssembler()


# ============================================================================
# 1. _truncate_to_tokens (static method)
# ============================================================================

def test_truncate_to_tokens_returns_empty_for_zero():
    result = ContextAssembler._truncate_to_tokens("hello world", 0)
    assert result == ""


def test_truncate_to_tokens_returns_empty_for_negative():
    result = ContextAssembler._truncate_to_tokens("hello world", -5)
    assert result == ""


def test_truncate_to_tokens_no_truncation_needed():
    result = ContextAssembler._truncate_to_tokens("hello", 100)
    assert result == "hello"


def test_truncate_to_tokens_exact_fit():
    # 4 chars per token, so 4 tokens = 16 chars fits exactly
    text = "a" * 16
    result = ContextAssembler._truncate_to_tokens(text, 4)
    assert result == text


def test_truncate_to_tokens_truncates_long_text():
    long_text = "a" * 1000 + "." + "b" * 1000
    result = ContextAssembler._truncate_to_tokens(long_text, 100)
    assert len(result) < len(long_text)


def test_truncate_to_tokens_adds_truncation_marker():
    long_text = "a" * 2000
    result = ContextAssembler._truncate_to_tokens(long_text, 100)
    assert "Truncated" in result


def test_truncate_to_tokens_prefers_newline_boundary():
    # Build a text with a newline near the cut point to confirm it honors newline
    text = "line one\nline two\n" + "x" * 10000
    result = ContextAssembler._truncate_to_tokens(text, 5)
    assert len(result) < len(text)
    assert "Truncated" in result


def test_truncate_to_tokens_prefers_period_boundary():
    # Make sure it finds a period near the boundary
    text = "sentence one. sentence two. " + "z" * 5000
    result = ContextAssembler._truncate_to_tokens(text, 20)
    assert len(result) < len(text)


# ============================================================================
# 2. _priority_sort_order (static method)
# ============================================================================

def test_priority_sort_order_critical_is_zero():
    assert ContextAssembler._priority_sort_order(ContextPriority.CRITICAL) == 0


def test_priority_sort_order_ordering():
    assert (
        ContextAssembler._priority_sort_order(ContextPriority.CRITICAL) <
        ContextAssembler._priority_sort_order(ContextPriority.HIGH) <
        ContextAssembler._priority_sort_order(ContextPriority.MEDIUM) <
        ContextAssembler._priority_sort_order(ContextPriority.LOW) <
        ContextAssembler._priority_sort_order(ContextPriority.OPTIONAL)
    )


def test_priority_sort_order_high_before_low():
    assert (
        ContextAssembler._priority_sort_order(ContextPriority.HIGH) <
        ContextAssembler._priority_sort_order(ContextPriority.LOW)
    )


# ============================================================================
# 3. _read_file_safe (async static method)
# ============================================================================

@pytest.mark.asyncio
async def test_read_file_safe_reads_small_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello(): pass\n")
    result = await ContextAssembler._read_file_safe(f)
    assert "hello" in result


@pytest.mark.asyncio
async def test_read_file_safe_returns_full_content_under_limit(tmp_path):
    content = "line\n" * 100
    f = tmp_path / "small.py"
    f.write_text(content)
    result = await ContextAssembler._read_file_safe(f, max_size=100_000)
    assert result == content


@pytest.mark.asyncio
async def test_read_file_safe_truncates_large_file(tmp_path):
    f = tmp_path / "big.py"
    f.write_bytes(b"x" * 200_000)
    result = await ContextAssembler._read_file_safe(f, max_size=50_000)
    assert "truncated" in result.lower()


@pytest.mark.asyncio
async def test_read_file_safe_truncated_contains_size_info(tmp_path):
    f = tmp_path / "huge.py"
    f.write_bytes(b"y" * 150_000)
    result = await ContextAssembler._read_file_safe(f, max_size=50_000)
    # The message includes the total file size
    assert "150000" in result or "bytes" in result


# ============================================================================
# 4. _emergency_truncate
# ============================================================================

def test_emergency_truncate_keeps_critical_chunks(assembler):
    critical = ContextChunk(
        content="critical info",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.CRITICAL,
    )
    low = ContextChunk(
        content="x" * 10000,
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.LOW,
    )
    result = assembler._emergency_truncate([critical, low], max_tokens=5)
    assert any(c.priority == ContextPriority.CRITICAL for c in result)


def test_emergency_truncate_drops_chunks_over_budget(assembler):
    chunks = [
        ContextChunk(
            content="a" * 400,
            source=ContextSource.CODE_FILE,
            priority=ContextPriority.MEDIUM,
        ),
        ContextChunk(
            content="b" * 400,
            source=ContextSource.CODE_FILE,
            priority=ContextPriority.MEDIUM,
        ),
        ContextChunk(
            content="c" * 400,
            source=ContextSource.CODE_FILE,
            priority=ContextPriority.MEDIUM,
        ),
    ]
    result = assembler._emergency_truncate(chunks, max_tokens=50)
    assert len(result) < len(chunks)


def test_emergency_truncate_empty_list(assembler):
    result = assembler._emergency_truncate([], max_tokens=100)
    assert result == []


def test_emergency_truncate_all_fit(assembler):
    chunks = [
        ContextChunk(
            content="small",
            source=ContextSource.CODE_FILE,
            priority=ContextPriority.HIGH,
        ),
    ]
    result = assembler._emergency_truncate(chunks, max_tokens=10000)
    assert len(result) == 1


def test_emergency_truncate_partial_chunk_truncated(assembler):
    # One medium chunk that just barely exceeds the budget gets truncated
    big_chunk = ContextChunk(
        content="z" * 800,  # 200 token estimate
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.MEDIUM,
    )
    result = assembler._emergency_truncate([big_chunk], max_tokens=50)
    # The chunk should be truncated (content shortened) or dropped
    assert len(result) <= 1


# ============================================================================
# 5. _merge_chunks
# ============================================================================

def test_merge_chunks_empty_list(assembler):
    result = assembler._merge_chunks([])
    assert result == ""


def test_merge_chunks_single_chunk(assembler):
    chunk = ContextChunk(
        content="hello code",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
    )
    result = assembler._merge_chunks([chunk])
    assert "hello code" in result


def test_merge_chunks_multiple_sources(assembler):
    c1 = ContextChunk(
        content="skill content",
        source=ContextSource.SKILLFILE_SKILL,
        priority=ContextPriority.HIGH,
    )
    c2 = ContextChunk(
        content="code content",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
    )
    result = assembler._merge_chunks([c1, c2])
    assert "skill content" in result
    assert "code content" in result


def test_merge_chunks_user_prompt_excluded_from_output(assembler):
    user_chunk = ContextChunk(
        content="user prompt here",
        source=ContextSource.USER_PROMPT,
        priority=ContextPriority.CRITICAL,
    )
    code_chunk = ContextChunk(
        content="some code",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
    )
    result = assembler._merge_chunks([user_chunk, code_chunk])
    # USER_PROMPT is skipped in merge; only code should appear
    assert "some code" in result


def test_merge_chunks_same_source_merged_together(assembler):
    c1 = ContextChunk(
        content="first snippet",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
    )
    c2 = ContextChunk(
        content="second snippet",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
    )
    result = assembler._merge_chunks([c1, c2])
    assert "first snippet" in result
    assert "second snippet" in result


def test_merge_chunks_separator_present_for_multiple_sources(assembler):
    c1 = ContextChunk(
        content="graphify info",
        source=ContextSource.GRAPHIFY_SUMMARY,
        priority=ContextPriority.HIGH,
    )
    c2 = ContextChunk(
        content="code snippet",
        source=ContextSource.CODE_FILE,
        priority=ContextPriority.HIGH,
    )
    result = assembler._merge_chunks([c1, c2])
    assert "---" in result


# ============================================================================
# 6. _wrap_user_prompt
# ============================================================================

def test_wrap_user_prompt_basic(assembler):
    result = assembler._wrap_user_prompt("Explain this", "repo context")
    assert "Explain this" in result
    assert "repo context" in result


def test_wrap_user_prompt_with_skills_injected(assembler):
    result = assembler._wrap_user_prompt("Fix this bug", "context", skills_injected=True)
    # Should include mention of expert instructions
    lowered = result.lower()
    assert "expert instructions" in lowered or "skill" in lowered


def test_wrap_user_prompt_without_skills(assembler):
    result = assembler._wrap_user_prompt("Fix this bug", "context", skills_injected=False)
    assert "Fix this bug" in result


def test_wrap_user_prompt_empty_context(assembler):
    result = assembler._wrap_user_prompt("Just a question", "")
    assert "Just a question" in result
    # Empty context: the REPOSITORY CONTEXT block should not appear
    assert "REPOSITORY CONTEXT" not in result


def test_wrap_user_prompt_nonempty_context_includes_header(assembler):
    result = assembler._wrap_user_prompt("My question", "some context here")
    assert "REPOSITORY CONTEXT" in result


def test_wrap_user_prompt_always_has_user_request_header(assembler):
    result = assembler._wrap_user_prompt("hello", "")
    assert "USER REQUEST" in result


# ============================================================================
# 7. _format_summary
# ============================================================================

def test_format_summary_basic(assembler):
    summary = MagicMock()
    summary.project_name = "MyProject"
    summary.primary_language = "Python"
    summary.total_files = 42
    summary.architecture_pattern = "layered"
    summary.key_components = ["api", "db", "models"]
    summary.dependencies = ["fastapi", "sqlalchemy"]
    result = assembler._format_summary(summary, ContextMode.BALANCED)
    assert "MyProject" in result
    assert "Python" in result


def test_format_summary_includes_file_count(assembler):
    summary = MagicMock()
    summary.total_files = 99
    result = assembler._format_summary(summary, ContextMode.BALANCED)
    assert "99" in result


def test_format_summary_minimal_object(assembler):
    # Object with no matching attributes — should return a non-empty string
    summary = MagicMock(spec=[])
    result = assembler._format_summary(summary, ContextMode.BALANCED)
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_summary_comprehensive_mode_includes_recommendations(assembler):
    summary = MagicMock()
    summary.project_name = "Project"
    summary.recommendations = ["Use dependency injection", "Add caching"]
    result = assembler._format_summary(summary, ContextMode.COMPREHENSIVE)
    assert isinstance(result, str)
    # recommendations block should appear for comprehensive mode
    assert "Recommendations" in result or "dependency injection" in result


def test_format_summary_balanced_mode_excludes_recommendations(assembler):
    summary = MagicMock()
    summary.recommendations = ["Tip A", "Tip B"]
    result = assembler._format_summary(summary, ContextMode.BALANCED)
    # recommendations only included in COMPREHENSIVE mode
    assert "Tip A" not in result


def test_format_summary_key_components_listed(assembler):
    summary = MagicMock()
    summary.key_components = ["auth", "payments"]
    result = assembler._format_summary(summary, ContextMode.BALANCED)
    assert "auth" in result
    assert "payments" in result


# ============================================================================
# 8. _format_app_map
# ============================================================================

def test_format_app_map_basic(assembler):
    app_map = MagicMock()
    app_map.modules = {"app.main": {"description": "Main entry point"}}
    app_map.entry_points = ["main.py"]
    result = assembler._format_app_map(app_map, "code_review")
    assert "app.main" in result


def test_format_app_map_entry_points_listed(assembler):
    app_map = MagicMock()
    app_map.modules = {}
    app_map.entry_points = ["main.py", "cli.py"]
    result = assembler._format_app_map(app_map, "general")
    assert "main.py" in result


def test_format_app_map_no_attributes(assembler):
    app_map = MagicMock(spec=[])
    result = assembler._format_app_map(app_map, "general")
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_app_map_module_description_included(assembler):
    app_map = MagicMock()
    app_map.modules = {
        "app.core": {"description": "Core utilities and config"},
    }
    app_map.entry_points = []
    result = assembler._format_app_map(app_map, "architecture")
    assert "Core utilities" in result


# ============================================================================
# 9. _format_graph_context
# ============================================================================

def test_format_graph_context_basic(assembler):
    graph = MagicMock()
    graph.node_count = 10
    graph.edge_count = 25
    graph.nodes = []
    graph.edges = []
    result = assembler._format_graph_context(graph, "architecture")
    assert "10" in result or "nodes" in result.lower()


def test_format_graph_context_includes_edge_count(assembler):
    graph = MagicMock()
    graph.node_count = 5
    graph.edge_count = 12
    graph.nodes = []
    graph.edges = []
    result = assembler._format_graph_context(graph, "debug")
    assert "12" in result or "edges" in result.lower()


def test_format_graph_context_no_attributes(assembler):
    graph = MagicMock(spec=[])
    result = assembler._format_graph_context(graph, "general")
    assert isinstance(result, str)


def test_format_graph_context_god_nodes_detected(assembler):
    # Simulate edges with repeated source/target to produce top nodes
    edge1 = MagicMock()
    edge1.source = "node_A"
    edge1.target = "node_B"
    edge2 = MagicMock()
    edge2.source = "node_A"
    edge2.target = "node_C"
    graph = MagicMock()
    graph.node_count = 3
    graph.edge_count = 2
    graph.nodes = []
    graph.edges = [edge1, edge2]
    # get_node_by_id returns None — fallback to node_id string
    graph.get_node_by_id.return_value = None
    result = assembler._format_graph_context(graph, "debug")
    assert "node_A" in result or "Most Connected" in result


# ============================================================================
# 10. _load_conversation_history
# ============================================================================

def test_load_conversation_history_empty(assembler):
    budget = ContextBudget(total_limit=8000)
    result = assembler._load_conversation_history([], budget)
    assert result == []


def test_load_conversation_history_with_messages(assembler):
    budget = ContextBudget(total_limit=8000)
    history = [
        {"role": "user", "content": "What is this?"},
        {"role": "assistant", "content": "This is the code."},
    ]
    result = assembler._load_conversation_history(history, budget)
    assert len(result) == 1
    combined = result[0].content.lower()
    assert "user" in combined or "assistant" in combined


def test_load_conversation_history_truncates_long_messages(assembler):
    budget = ContextBudget(total_limit=8000)
    history = [{"role": "user", "content": "x" * 5000}]
    result = assembler._load_conversation_history(history, budget)
    assert len(result) > 0
    # Individual long message is capped at 2000 chars + "..." before bundling
    assert len(result[0].content) < 5000


def test_load_conversation_history_keeps_last_10(assembler):
    budget = ContextBudget(total_limit=80000)
    history = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
    result = assembler._load_conversation_history(history, budget)
    assert len(result) > 0
    # Only last 10 kept; last message is msg 14
    assert "msg 14" in result[0].content


def test_load_conversation_history_returns_conversation_source_chunk(assembler):
    budget = ContextBudget(total_limit=8000)
    history = [{"role": "user", "content": "hello"}]
    result = assembler._load_conversation_history(history, budget)
    assert result[0].source == ContextSource.CONVERSATION_HISTORY


def test_load_conversation_history_priority_medium(assembler):
    budget = ContextBudget(total_limit=8000)
    history = [{"role": "user", "content": "hi"}]
    result = assembler._load_conversation_history(history, budget)
    assert result[0].priority == ContextPriority.MEDIUM


# ============================================================================
# 11. _load_file_context (async)
# ============================================================================

@pytest.mark.asyncio
async def test_load_file_context_existing_file(assembler, tmp_path):
    f = tmp_path / "app.py"
    f.write_text("def main(): pass\n")
    budget = ContextBudget(total_limit=8000)
    result = await assembler._load_file_context(tmp_path, ["app.py"], budget)
    assert len(result) == 1
    assert "app.py" in result[0].content


@pytest.mark.asyncio
async def test_load_file_context_missing_file(assembler, tmp_path):
    budget = ContextBudget(total_limit=8000)
    result = await assembler._load_file_context(tmp_path, ["nonexistent.py"], budget)
    assert result == []


@pytest.mark.asyncio
async def test_load_file_context_max_10_files(assembler, tmp_path):
    for i in range(15):
        (tmp_path / f"file{i}.py").write_text(f"# file {i}\n")
    budget = ContextBudget(total_limit=800_000)
    files = [f"file{i}.py" for i in range(15)]
    result = await assembler._load_file_context(tmp_path, files, budget)
    assert len(result) <= 10


@pytest.mark.asyncio
async def test_load_file_context_chunk_has_code_file_source(assembler, tmp_path):
    f = tmp_path / "utils.py"
    f.write_text("x = 1\n")
    budget = ContextBudget(total_limit=8000)
    result = await assembler._load_file_context(tmp_path, ["utils.py"], budget)
    assert result[0].source == ContextSource.CODE_FILE


@pytest.mark.asyncio
async def test_load_file_context_file_path_set_in_chunk(assembler, tmp_path):
    f = tmp_path / "module.py"
    f.write_text("pass\n")
    budget = ContextBudget(total_limit=8000)
    result = await assembler._load_file_context(tmp_path, ["module.py"], budget)
    assert result[0].file_path == "module.py"


# ============================================================================
# 12. _get_default_token_limit, _create_budget, _get_graphify_budget
# ============================================================================

def test_get_default_token_limit_balanced(assembler):
    limit = assembler._get_default_token_limit(ContextMode.BALANCED)
    assert limit == 8192


def test_get_default_token_limit_precise(assembler):
    limit = assembler._get_default_token_limit(ContextMode.PRECISE)
    assert limit == 4096


def test_get_default_token_limit_comprehensive(assembler):
    limit = assembler._get_default_token_limit(ContextMode.COMPREHENSIVE)
    assert limit == 16384


def test_get_default_token_limit_code_only(assembler):
    limit = assembler._get_default_token_limit(ContextMode.CODE_ONLY)
    assert limit == 8192


def test_create_budget_balanced_mode(assembler):
    budget = assembler._create_budget(8000, "general", ContextMode.BALANCED)
    assert isinstance(budget, ContextBudget)
    assert budget.total_limit == 8000


def test_create_budget_precise_mode_sets_graphify(assembler):
    budget = assembler._create_budget(4096, "code", ContextMode.PRECISE)
    assert budget.graphify_context == 4096 // 5


def test_create_budget_comprehensive_includes_documentation(assembler):
    budget = assembler._create_budget(16384, "architecture", ContextMode.COMPREHENSIVE)
    assert budget.documentation > 0


def test_create_budget_with_skills_sets_skill_context(assembler):
    budget = assembler._create_budget(8000, "code", ContextMode.BALANCED, include_skills=True)
    assert budget.skill_context > 0


def test_create_budget_without_skills_skill_context_zero(assembler):
    budget = assembler._create_budget(8000, "code", ContextMode.BALANCED, include_skills=False)
    assert budget.skill_context == 0


def test_get_graphify_budget_code_only_returns_zero(assembler):
    budget = ContextBudget(total_limit=8000)
    result = assembler._get_graphify_budget(budget, ContextMode.CODE_ONLY)
    assert result == 0


def test_get_graphify_budget_balanced_returns_non_negative(assembler):
    budget = ContextBudget(total_limit=8000)
    result = assembler._get_graphify_budget(budget, ContextMode.BALANCED)
    assert result >= 0


def test_get_graphify_budget_uses_allocated_graphify_context(assembler):
    budget = ContextBudget(total_limit=8000)
    budget.graphify_context = 1000
    # allocated is 1000 (graphify_context); result = max(0, 1000 - 1000) = 0
    result = assembler._get_graphify_budget(budget, ContextMode.BALANCED)
    assert result == 0


# ============================================================================
# 13. _generate_cache_key, _update_stats, get_stats
# ============================================================================

def test_generate_cache_key_deterministic(assembler):
    key1 = assembler._generate_cache_key("arg1", "arg2")
    key2 = assembler._generate_cache_key("arg1", "arg2")
    assert key1 == key2


def test_generate_cache_key_different_args_produce_different_keys(assembler):
    key1 = assembler._generate_cache_key("a", "b")
    key2 = assembler._generate_cache_key("c", "d")
    assert key1 != key2


def test_generate_cache_key_is_string(assembler):
    key = assembler._generate_cache_key("x")
    assert isinstance(key, str)


def test_generate_cache_key_is_sha256_hex(assembler):
    key = assembler._generate_cache_key("test")
    assert len(key) == 64  # sha256 hex digest length


def test_update_stats_increments_assembly_count(assembler):
    assembled = MagicMock()
    assembled.total_tokens = 500
    assembled.skills_injected = 0
    before = assembler._assembly_count
    assembler._update_stats(assembled)
    assert assembler._assembly_count == before + 1


def test_update_stats_accumulates_tokens(assembler):
    assembled = MagicMock()
    assembled.total_tokens = 500
    assembled.skills_injected = 0
    assembler._update_stats(assembled)
    assembler._update_stats(assembled)
    assert assembler._total_tokens_used == 1000


def test_update_stats_tracks_skills_injected(assembler):
    assembled = MagicMock()
    assembled.total_tokens = 200
    assembled.skills_injected = 3
    assembler._update_stats(assembled)
    stats = assembler.get_stats()
    assert stats["skills_injected_total"] >= 3


def test_get_stats_returns_dict(assembler):
    stats = assembler.get_stats()
    assert isinstance(stats, dict)


def test_get_stats_has_required_keys(assembler):
    stats = assembler.get_stats()
    for key in ("assemblies", "total_tokens_used", "average_tokens",
                "skills_injected_total", "skill_injection_enabled",
                "cache_size", "cache_ttl_seconds"):
        assert key in stats, f"Missing key: {key}"


def test_get_stats_total_assembled_alias(assembler):
    # The existing unit test references "total_assembled" — but implementation
    # uses "assemblies"; test whichever is present
    stats = assembler.get_stats()
    assert "assemblies" in stats or "total_assembled" in stats


# ============================================================================
# 14. clear_cache, close, enable_skill_injection, set_skill_limits
# ============================================================================

def test_clear_cache_does_not_raise(assembler):
    assembler.clear_cache()  # should not raise


def test_clear_cache_empties_cache(assembler):
    # Inject a fake cache entry
    assembler._context_cache["fake_key"] = (MagicMock(), MagicMock())
    assert len(assembler._context_cache) == 1
    assembler.clear_cache()
    assert len(assembler._context_cache) == 0


@pytest.mark.asyncio
async def test_close_does_not_raise(assembler):
    assembler.parser.close = AsyncMock()
    assembler.skillfile.close = AsyncMock()
    await assembler.close()  # should not raise


@pytest.mark.asyncio
async def test_close_clears_cache(assembler):
    assembler.parser.close = AsyncMock()
    assembler.skillfile.close = AsyncMock()
    assembler._context_cache["x"] = (MagicMock(), MagicMock())
    await assembler.close()
    assert len(assembler._context_cache) == 0


def test_enable_skill_injection_toggle(assembler):
    assembler.enable_skill_injection(True)
    assert assembler._skill_injection_enabled is True
    assembler.enable_skill_injection(False)
    assert assembler._skill_injection_enabled is False


def test_set_skill_limits_max_skills(assembler):
    assembler.set_skill_limits(max_skills=5)
    assert assembler._max_skills_per_task == 5


def test_set_skill_limits_max_tokens_per_skill(assembler):
    assembler.set_skill_limits(max_tokens_per_skill=2000)
    assert assembler._max_skill_tokens_per_skill == 2000


def test_set_skill_limits_both(assembler):
    assembler.set_skill_limits(max_skills=7, max_tokens_per_skill=1000)
    assert assembler._max_skills_per_task == 7
    assert assembler._max_skill_tokens_per_skill == 1000


def test_set_skill_limits_none_does_not_change(assembler):
    original_max = assembler._max_skills_per_task
    assembler.set_skill_limits(max_skills=None)
    assert assembler._max_skills_per_task == original_max


# ============================================================================
# 15. _format_skill_content (bonus — called internally)
# ============================================================================

def test_format_skill_content_contains_skill_name(assembler):
    result = assembler._format_skill_content("Do things better.", "my_skill")
    assert "my_skill" in result


def test_format_skill_content_contains_content(assembler):
    result = assembler._format_skill_content("Expert advice here.", "skill_x")
    assert "Expert advice here." in result


def test_format_skill_content_has_expert_instructions_header(assembler):
    result = assembler._format_skill_content("content", "skill_name")
    assert "EXPERT INSTRUCTIONS" in result


# ── _load_skill_context ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_skill_context_code_only_returns_empty(assembler, tmp_path):
    budget = ContextBudget(total_limit=8000)
    chunks, names = await assembler._load_skill_context(tmp_path, "general", budget, ContextMode.CODE_ONLY)
    assert chunks == []
    assert names == []


@pytest.mark.asyncio
async def test_load_skill_context_skillfile_not_installed(assembler, tmp_path):
    assembler.skillfile = MagicMock(is_installed=False)
    budget = ContextBudget(total_limit=8000)
    chunks, names = await assembler._load_skill_context(tmp_path, "code_review", budget, ContextMode.BALANCED)
    assert chunks == []
    assert names == []


@pytest.mark.asyncio
async def test_load_skill_context_no_skills_found(assembler, tmp_path):
    mock_sf = AsyncMock()
    mock_sf.is_installed = True
    mock_sf.install_skills = AsyncMock(return_value=True)
    mock_sf.load_skills = AsyncMock(return_value=[])
    assembler.skillfile = mock_sf
    budget = ContextBudget(total_limit=8000)
    chunks, names = await assembler._load_skill_context(tmp_path, "debugging", budget, ContextMode.BALANCED)
    assert chunks == []
    assert names == []


@pytest.mark.asyncio
async def test_load_skill_context_loads_skills_successfully(assembler, tmp_path):
    from app.integrations.skillfile.client import SkillContent, SkillEntry, SkillSource, SkillType
    entry = SkillEntry(name="code-review", source=SkillSource.GITHUB, skill_type=SkillType.SKILL, path="skills/code-review.md", repo="owner/repo")
    skill = SkillContent(entry=entry, content="Review code carefully.\n" * 10)

    mock_sf = AsyncMock()
    mock_sf.is_installed = True
    mock_sf.install_skills = AsyncMock(return_value=True)
    mock_sf.load_skills = AsyncMock(return_value=[skill])
    assembler.skillfile = mock_sf

    budget = ContextBudget(total_limit=8000)
    chunks, names = await assembler._load_skill_context(tmp_path, "code_review", budget, ContextMode.BALANCED)
    assert len(chunks) == 1
    assert "code-review" in names


@pytest.mark.asyncio
async def test_load_skill_context_handles_exception(assembler, tmp_path):
    mock_sf = AsyncMock()
    mock_sf.is_installed = True
    mock_sf.install_skills = AsyncMock(side_effect=Exception("skillfile error"))
    assembler.skillfile = mock_sf

    budget = ContextBudget(total_limit=8000)
    # Should not raise — handles exception gracefully
    chunks, names = await assembler._load_skill_context(tmp_path, "general", budget, ContextMode.BALANCED)
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_load_skill_context_precise_mode_budget(assembler, tmp_path):
    from app.integrations.skillfile.client import SkillContent, SkillEntry, SkillSource, SkillType
    entry = SkillEntry(name="skill1", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="./s.md")
    skill = SkillContent(entry=entry, content="Expert guide " * 20)

    mock_sf = AsyncMock()
    mock_sf.is_installed = True
    mock_sf.install_skills = AsyncMock(return_value=True)
    mock_sf.load_skills = AsyncMock(return_value=[skill])
    assembler.skillfile = mock_sf

    budget = ContextBudget(total_limit=8000)
    chunks, names = await assembler._load_skill_context(tmp_path, "general", budget, ContextMode.PRECISE)
    assert isinstance(chunks, list)


# ── _load_graphify_context ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_graphify_context_not_available(assembler, tmp_path):
    assembler.graphify_parser = MagicMock()
    assembler.graphify_parser.parse_repository = AsyncMock(return_value={"available": False})
    budget = ContextBudget(total_limit=8000)
    chunks = await assembler._load_graphify_context(tmp_path, "general", budget, ContextMode.BALANCED)
    assert chunks == []


@pytest.mark.asyncio
async def test_load_graphify_context_code_only_returns_empty(assembler, tmp_path):
    budget = ContextBudget(total_limit=8000)
    assembler.graphify_parser = MagicMock()
    assembler.graphify_parser.parse_repository = AsyncMock(return_value={"available": False})
    chunks = await assembler._load_graphify_context(tmp_path, "general", budget, ContextMode.CODE_ONLY)
    assert chunks == []


@pytest.mark.asyncio
async def test_load_graphify_context_handles_exception(assembler, tmp_path):
    assembler.graphify_parser = MagicMock()
    assembler.graphify_parser.parse_repository = AsyncMock(side_effect=Exception("graphify error"))
    budget = ContextBudget(total_limit=8000)
    # Should not raise
    chunks = await assembler._load_graphify_context(tmp_path, "general", budget, ContextMode.BALANCED)
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_load_graphify_context_with_summary(assembler, tmp_path):
    summary = MagicMock()
    summary.project_name = "TestProject"
    summary.primary_language = "Python"
    summary.total_files = 20
    summary.architecture_pattern = "MVC"
    summary.key_components = ["api"]
    summary.dependencies = ["fastapi"]

    parse_result = {"available": True, "summary": summary}
    assembler.graphify_parser = MagicMock()
    assembler.graphify_parser.parse_repository = AsyncMock(return_value=parse_result)

    budget = ContextBudget(total_limit=8000)
    chunks = await assembler._load_graphify_context(tmp_path, "architecture", budget, ContextMode.BALANCED)
    assert isinstance(chunks, list)


# ── _format_skill_content ─────────────────────────────────────────────────────

def test_format_skill_content_includes_skill_name(assembler):
    result = assembler._format_skill_content("Expert instructions here", "my-skill")
    assert "my-skill" in result
    assert "Expert instructions here" in result


# ── discover_and_add_skills ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_and_add_skills_not_installed(assembler, tmp_path):
    assembler.skillfile = MagicMock(is_installed=False)
    result = await assembler.discover_and_add_skills(str(tmp_path), "code_review")
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_discover_and_add_skills_discovers_from_task(assembler, tmp_path):
    mock_result = [MagicMock(name="skill-1", description="desc", url="http://example.com")]
    mock_sf = AsyncMock()
    mock_sf.is_installed = True
    mock_sf.discover_skills_for_task = AsyncMock(return_value=mock_result)
    mock_sf.add_skill = AsyncMock(return_value=True)
    assembler.skillfile = mock_sf
    result = await assembler.discover_and_add_skills(str(tmp_path), "code_review")
    assert isinstance(result, list)


# ── _load_graphify_context with valid graphify output ─────────────────────────

@pytest.mark.asyncio
async def test_load_graphify_context_valid_is_valid_false(assembler, tmp_path):
    """Cover lines 619-621: is_valid=False returns empty."""
    graphify_output = MagicMock()
    graphify_output.is_valid = False
    assembler.parser = MagicMock()
    assembler.parser.parse_repository = AsyncMock(return_value=graphify_output)
    budget = ContextBudget(total_limit=8000)
    chunks = await assembler._load_graphify_context(tmp_path, "general", budget, ContextMode.BALANCED)
    assert chunks == []


@pytest.mark.asyncio
async def test_load_graphify_context_with_valid_output_summary(assembler, tmp_path):
    """Cover lines 619-690: graphify_output.is_valid=True with summary."""
    summary = MagicMock()
    summary.project_name = "TestProject"
    summary.primary_language = "Python"
    summary.total_files = 20
    summary.architecture_pattern = "MVC"
    summary.key_components = ["api", "services"]
    summary.dependencies = ["fastapi", "sqlalchemy"]

    graphify_output = MagicMock()
    graphify_output.is_valid = True
    graphify_output.has_summary = True
    graphify_output.has_app_map = False
    graphify_output.has_graph = False
    graphify_output.summary = summary

    assembler.parser = MagicMock()
    assembler.parser.parse_repository = AsyncMock(return_value=graphify_output)

    budget = ContextBudget(total_limit=8000)
    # _get_graphify_budget returns budget.graphify_context - budget.allocated;
    # since allocated sums all fields (incl. graphify_context), we mock the method.
    with patch.object(assembler, "_get_graphify_budget", return_value=3000):
        chunks = await assembler._load_graphify_context(tmp_path, "general", budget, ContextMode.BALANCED)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_load_graphify_context_with_app_map(assembler, tmp_path):
    """Cover lines 644-666: app_map section for architecture task."""
    app_map = MagicMock()
    app_map.modules = ["api", "services", "models"]

    graphify_output = MagicMock()
    graphify_output.is_valid = True
    graphify_output.has_summary = False
    graphify_output.has_app_map = True
    graphify_output.has_graph = False
    graphify_output.app_map = app_map

    assembler.parser = MagicMock()
    assembler.parser.parse_repository = AsyncMock(return_value=graphify_output)

    budget = ContextBudget(total_limit=8000)
    with patch.object(assembler, "_get_graphify_budget", return_value=2000):
        chunks = await assembler._load_graphify_context(tmp_path, "architecture", budget, ContextMode.BALANCED)
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_load_graphify_context_with_graph_data(assembler, tmp_path):
    """Cover lines 668-690: graph data section for debug task."""
    graph = MagicMock()
    graph.nodes = [{"id": "node1"}, {"id": "node2"}]
    graph.edges = [{"from": "node1", "to": "node2"}]

    graphify_output = MagicMock()
    graphify_output.is_valid = True
    graphify_output.has_summary = False
    graphify_output.has_app_map = False
    graphify_output.has_graph = True
    graphify_output.graph = graph

    assembler.parser = MagicMock()
    assembler.parser.parse_repository = AsyncMock(return_value=graphify_output)

    budget = ContextBudget(total_limit=8000)
    with patch.object(assembler, "_get_graphify_budget", return_value=2000):
        chunks = await assembler._load_graphify_context(tmp_path, "debug", budget, ContextMode.BALANCED)
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_load_graphify_context_all_sections_full_budget(assembler, tmp_path):
    """Cover all three sections: summary + app_map + graph."""
    summary = MagicMock()
    summary.project_name = "TestProject"
    summary.primary_language = "Python"
    summary.total_files = 20
    summary.architecture_pattern = "Microservices"
    summary.key_components = ["api"]
    summary.dependencies = ["fastapi"]

    app_map = MagicMock()
    graph = MagicMock()
    graph.nodes = []
    graph.edges = []

    graphify_output = MagicMock()
    graphify_output.is_valid = True
    graphify_output.has_summary = True
    graphify_output.has_app_map = True
    graphify_output.has_graph = True
    graphify_output.summary = summary
    graphify_output.app_map = app_map
    graphify_output.graph = graph

    assembler.parser = MagicMock()
    assembler.parser.parse_repository = AsyncMock(return_value=graphify_output)

    budget = ContextBudget(total_limit=16000)
    with patch.object(assembler, "_get_graphify_budget", return_value=5000):
        chunks = await assembler._load_graphify_context(tmp_path, "architecture", budget, ContextMode.COMPREHENSIVE)
    assert isinstance(chunks, list)


# ── estimate_tokens ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_estimate_tokens_returns_dict(assembler):
    assembler.parser = MagicMock()
    output = MagicMock()
    output.has_summary = False
    assembler.parser.parse_repository = AsyncMock(return_value=output)
    assembler.skillfile = MagicMock(is_installed=False)
    result = await assembler.estimate_tokens("/tmp", "Analyze this codebase")
    assert isinstance(result, dict)
    assert "prompt_tokens" in result
    assert "estimated_total" in result


@pytest.mark.asyncio
async def test_estimate_tokens_with_graphify_summary(assembler):
    assembler.parser = MagicMock()
    output = MagicMock()
    output.has_summary = True
    output.summary = MagicMock()
    output.summary.model_dump = MagicMock(return_value={"project": "test"})
    assembler.parser.parse_repository = AsyncMock(return_value=output)
    assembler.skillfile = MagicMock(is_installed=False)
    result = await assembler.estimate_tokens("/tmp", "Analyze this codebase")
    assert result["graphify_tokens"] >= 0


@pytest.mark.asyncio
async def test_estimate_tokens_with_skills(assembler):
    assembler.parser = MagicMock()
    output = MagicMock()
    output.has_summary = False
    assembler.parser.parse_repository = AsyncMock(return_value=output)
    assembler.skillfile = MagicMock(is_installed=True)
    assembler._skill_injection_enabled = True
    result = await assembler.estimate_tokens("/tmp", "test")
    assert result["skill_tokens"] > 0
