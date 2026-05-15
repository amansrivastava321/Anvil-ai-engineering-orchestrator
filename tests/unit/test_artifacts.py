import json
import pytest
from pathlib import Path
from app.artifacts.store import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(base_path=str(tmp_path / "artifacts"))


# ── save_run / get_run ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_run_creates_files(store):
    run_id = await store.save_run({
        "prompt": "hello world",
        "response": "hi there",
        "repo_path": "/tmp/repo",
        "workflow_type": "general_qa",
        "status": "completed",
    })
    assert run_id is not None
    run_dir = store.runs_path / run_id
    assert (run_dir / "run.json").exists()
    assert (run_dir / "prompt.md").exists()
    assert (run_dir / "response.md").exists()


@pytest.mark.asyncio
async def test_save_run_uses_provided_run_id(store):
    run_id = await store.save_run({"run_id": "my-run-123", "prompt": "p"})
    assert run_id == "my-run-123"
    assert (store.runs_path / "my-run-123" / "run.json").exists()


@pytest.mark.asyncio
async def test_save_run_writes_optional_files(store):
    run_id = await store.save_run({
        "context": {"key": "val"},
        "logs": ["line1", "line2"],
        "test_results": {"passed": 5},
        "patches": "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n",
    })
    run_dir = store.runs_path / run_id
    assert (run_dir / "context.json").exists()
    assert (run_dir / "logs.txt").exists()
    assert (run_dir / "test_results.json").exists()
    assert (run_dir / "patches.diff").exists()


@pytest.mark.asyncio
async def test_get_run_returns_data(store):
    run_id = await store.save_run({"prompt": "test prompt", "status": "done"})
    run = await store.get_run(run_id)
    assert run is not None
    assert run["prompt"] == "test prompt"
    assert run["run_id"] == run_id


@pytest.mark.asyncio
async def test_get_run_missing_returns_none(store):
    result = await store.get_run("nonexistent-run-id")
    assert result is None


# ── list_runs ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_runs_returns_summaries(store):
    await store.save_run({"prompt": "first", "repo_path": "/repo"})
    await store.save_run({"prompt": "second", "repo_path": "/repo"})
    runs = await store.list_runs()
    assert len(runs) == 2
    # Each summary has run_id and truncated prompt
    for r in runs:
        assert "run_id" in r
        assert "prompt" in r


@pytest.mark.asyncio
async def test_list_runs_filters_by_repo_path(store):
    await store.save_run({"prompt": "a", "repo_path": "/repo-a"})
    await store.save_run({"prompt": "b", "repo_path": "/repo-b"})
    runs = await store.list_runs(repo_path="/repo-a")
    assert len(runs) == 1
    assert runs[0]["prompt"] == "a"


@pytest.mark.asyncio
async def test_list_runs_empty_store(store):
    runs = await store.list_runs()
    assert runs == []


# ── search_runs ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_runs_matches_prompt(store):
    await store.save_run({"prompt": "refactor the auth module", "response": "done"})
    await store.save_run({"prompt": "fix the login bug", "response": "done"})
    results = await store.search_runs("auth")
    assert len(results) == 1
    assert "auth" in results[0]["prompt"]


@pytest.mark.asyncio
async def test_search_runs_matches_response(store):
    await store.save_run({"prompt": "do something", "response": "applied auth patch"})
    results = await store.search_runs("auth patch")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_runs_no_match(store):
    await store.save_run({"prompt": "hello", "response": "world"})
    results = await store.search_runs("zzznomatch")
    assert results == []


# ── legacy compat ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_artifact_legacy_returns_run_id(store):
    result = await store.save_artifact(
        execution_id="exec-1",
        artifact_type="response",
        content="hello world",
    )
    assert result == "exec-1"


@pytest.mark.asyncio
async def test_get_artifact_legacy_returns_run_data(store):
    await store.save_artifact("exec-2", "response", "content here")
    data = await store.get_artifact("exec-2", "response")
    assert data is not None
    assert data["run_id"] == "exec-2"


@pytest.mark.asyncio
async def test_list_artifacts_legacy_returns_list(store):
    await store.save_artifact("exec-3", "prompt", "p")
    artifacts = await store.list_artifacts("exec-3")
    assert len(artifacts) == 1
    assert artifacts[0]["run_id"] == "exec-3"


@pytest.mark.asyncio
async def test_list_artifacts_legacy_empty_for_unknown(store):
    artifacts = await store.list_artifacts("unknown-exec")
    assert artifacts == []
