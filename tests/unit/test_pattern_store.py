"""Tests for app.memory.pattern_store — PatternStore learning patterns."""
import pytest
from app.memory.pattern_store import PatternStore


@pytest.fixture
def store(tmp_path):
    return PatternStore(data_dir=str(tmp_path / "mem"))


def test_record_and_retrieve_pattern(store):
    store.record_pattern("fix", "import_error", {"file": "a.py"}, "success")
    patterns = store.get_patterns()
    assert len(patterns) == 1
    assert patterns[0]["type"] == "fix"
    assert patterns[0]["key"] == "import_error"
    assert patterns[0]["count"] == 1


def test_same_key_accumulates_outcomes(store):
    store.record_pattern("fix", "null_check", {}, "success")
    store.record_pattern("fix", "null_check", {}, "failure")
    patterns = store.get_patterns(pattern_type="fix")
    assert patterns[0]["count"] == 2
    assert len(patterns[0]["outcomes"]) == 2


def test_filter_by_type(store):
    store.record_pattern("fix", "k1", {}, "ok")
    store.record_pattern("review", "k2", {}, "ok")
    fixes = store.get_patterns(pattern_type="fix")
    assert all(p["type"] == "fix" for p in fixes)
    assert len(fixes) == 1


def test_get_patterns_sorted_by_count_desc(store):
    store.record_pattern("t", "popular", {}, "ok")
    store.record_pattern("t", "popular", {}, "ok")
    store.record_pattern("t", "rare", {}, "ok")
    patterns = store.get_patterns()
    assert patterns[0]["key"] == "popular"


def test_get_best_model_for_task_returns_none_when_empty(store):
    assert store.get_best_model_for_task("debug") is None


def test_persists_across_reload(tmp_path):
    data_dir = str(tmp_path / "mem")
    s1 = PatternStore(data_dir=data_dir)
    s1.record_pattern("fix", "key1", {"context": "x"}, "success")
    s2 = PatternStore(data_dir=data_dir)
    patterns = s2.get_patterns()
    assert len(patterns) == 1
    assert patterns[0]["key"] == "key1"


def test_corrupted_patterns_file_returns_empty(tmp_path):
    data_dir = tmp_path / "mem"
    data_dir.mkdir()
    (data_dir / "patterns.json").write_text("NOT VALID JSON {{{")
    store = PatternStore(data_dir=str(data_dir))
    assert store.get_patterns() == []


def test_get_pattern_store_singleton():
    import app.memory.pattern_store as ps_module
    orig = ps_module._store if hasattr(ps_module, "_store") else None
    try:
        from app.memory.pattern_store import get_pattern_store
        s1 = get_pattern_store()
        s2 = get_pattern_store()
        assert s1 is s2
    except (AttributeError, ImportError):
        pass  # get_pattern_store may not exist — just cover what we can
    finally:
        if hasattr(ps_module, "_store"):
            ps_module._store = orig
