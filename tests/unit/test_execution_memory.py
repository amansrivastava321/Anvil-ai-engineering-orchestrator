"""Tests for app.memory.execution_memory — ExecutionMemory run tracking."""
import pytest
from app.memory.execution_memory import ExecutionMemory


@pytest.fixture
def mem(tmp_path):
    return ExecutionMemory(data_dir=str(tmp_path / "mem"))


def test_record_and_get_recent(mem):
    mem.record("run-1", "debug", "qwen2.5-coder:7b", "debug", True, 1234.0)
    recent = mem.get_recent(limit=5)
    assert len(recent) == 1
    assert recent[0]["run_id"] == "run-1"
    assert recent[0]["success"] is True


def test_get_recent_respects_limit(mem):
    for i in range(10):
        mem.record(f"run-{i}", "debug", "model", "debug", True, 100.0)
    assert len(mem.get_recent(limit=3)) == 3


def test_get_model_stats_no_data(mem):
    stats = mem.get_model_stats("unknown-model")
    assert stats["runs"] == 0


def test_get_model_stats_with_data(mem):
    mem.record("r1", "debug", "qwen", "debug", True, 100.0)
    mem.record("r2", "debug", "qwen", "debug", False, 200.0)
    stats = mem.get_model_stats("qwen")
    assert stats["runs"] == 2
    assert stats["success_rate"] == 0.5
    assert stats["avg_duration_ms"] == 150.0


def test_get_workflow_stats_no_data(mem):
    stats = mem.get_workflow_stats("nonexistent")
    assert stats["runs"] == 0


def test_get_workflow_stats_with_data(mem):
    mem.record("r1", "audit", "model", "audit", True, 500.0)
    mem.record("r2", "audit", "model", "audit", True, 600.0)
    stats = mem.get_workflow_stats("audit")
    assert stats["runs"] == 2
    assert stats["success_rate"] == 1.0


def test_record_with_optional_metadata(mem):
    mem.record("r1", "wf", "model", "type", True, 100.0, metadata={"files": 3})
    recent = mem.get_recent()
    assert recent[-1]["metadata"] == {"files": 3}


def test_record_with_error_field(mem):
    mem.record("r1", "wf", "model", "type", False, 0.0, error="timeout")
    recent = mem.get_recent()
    assert recent[-1]["error"] == "timeout"


def test_persists_across_reload(tmp_path):
    data_dir = str(tmp_path / "mem")
    m1 = ExecutionMemory(data_dir=data_dir)
    m1.record("persist-1", "wf", "model", "type", True, 100.0)
    m2 = ExecutionMemory(data_dir=data_dir)
    assert len(m2.get_recent()) == 1
    assert m2.get_recent()[0]["run_id"] == "persist-1"


def test_capped_at_1000_entries(mem):
    for i in range(1010):
        mem.record(f"run-{i}", "wf", "model", "type", True, 100.0)
    assert len(mem.get_recent(limit=2000)) == 1000


def test_load_corrupted_json_returns_empty(tmp_path):
    data_dir = tmp_path / "mem"
    data_dir.mkdir()
    (data_dir / "executions.json").write_text("NOT_VALID_JSON{{{")
    mem = ExecutionMemory(data_dir=str(data_dir))
    assert mem.get_recent() == []


def test_get_execution_memory_singleton(tmp_path):
    import app.memory.execution_memory as em_module
    orig = em_module._memory
    em_module._memory = None
    try:
        from app.memory.execution_memory import get_execution_memory
        m1 = get_execution_memory()
        m2 = get_execution_memory()
        assert m1 is m2
    finally:
        em_module._memory = orig
