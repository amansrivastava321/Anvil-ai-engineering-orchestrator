"""Tests for app.memory.memory_store — MemoryStore key-value persistence."""
import pytest
from app.memory.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(store_path=str(tmp_path / "memory.json"))


def test_set_and_get_string(store):
    store.set("greeting", "hello")
    assert store.get("greeting") == "hello"


def test_set_and_get_dict(store):
    store.set("config", {"k": "v", "n": 42})
    assert store.get("config") == {"k": "v", "n": 42}


def test_get_missing_key_returns_default(store):
    assert store.get("missing") is None
    assert store.get("missing", "fallback") == "fallback"


def test_delete_removes_key(store):
    store.set("x", 1)
    store.delete("x")
    assert store.get("x") is None


def test_delete_missing_key_is_safe(store):
    store.delete("never_existed")  # must not raise


def test_keys_lists_all_stored_keys(store):
    store.set("a", 1)
    store.set("b", 2)
    assert set(store.keys()) == {"a", "b"}


def test_all_returns_values_without_metadata(store):
    store.set("alpha", "A")
    store.set("beta", "B")
    data = store.all()
    assert data == {"alpha": "A", "beta": "B"}


def test_persisted_across_instance_reload(tmp_path):
    path = str(tmp_path / "mem.json")
    s1 = MemoryStore(store_path=path)
    s1.set("persistent", 99)
    s2 = MemoryStore(store_path=path)
    assert s2.get("persistent") == 99


def test_corrupted_file_loads_empty(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{{not valid json}}")
    store = MemoryStore(store_path=str(path))
    assert store.keys() == []


def test_overwrite_existing_key(store):
    store.set("counter", 1)
    store.set("counter", 2)
    assert store.get("counter") == 2


def test_get_memory_store_singleton():
    import app.memory.memory_store as ms_module
    orig = ms_module._store
    ms_module._store = None
    try:
        from app.memory.memory_store import get_memory_store
        m1 = get_memory_store()
        m2 = get_memory_store()
        assert m1 is m2
    finally:
        ms_module._store = orig
