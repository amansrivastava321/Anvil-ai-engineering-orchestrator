"""Tests for MonitorAgent, ProactiveService, and monitor API endpoints."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents.specialized.monitor_agent import (
    ChangeEvent,
    MonitorAgent,
    RepoStatus,
)
from app.services.proactive_service import ProactiveService, get_proactive_service


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama():
    client = MagicMock()
    client.default_model = "test-model"

    async def fake_chat(*args, **kwargs):
        yield "diagnosis result"

    client.chat = fake_chat
    return client


@pytest.fixture
def monitor_agent(mock_ollama):
    return MonitorAgent(ollama_client=mock_ollama)


@pytest.fixture
def sample_repo(tmp_path):
    """Real temporary git repo for filesystem tests."""
    import subprocess
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    (tmp_path / "hello.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    return tmp_path


# ─── ChangeEvent ──────────────────────────────────────────────────────────────

class TestChangeEvent:
    def test_str_representation(self):
        event = ChangeEvent(
            repo_path="/repo",
            commit_hash="abc1234567890",
            changed_files=["a.py", "b.py"],
            branch="main",
        )
        s = str(event)
        assert "abc12345" in s
        assert "2 files" in s
        assert "branch=main" in s

    def test_default_detected_at_is_utc(self):
        event = ChangeEvent(repo_path="/r", commit_hash="aaa", changed_files=[])
        assert event.detected_at.tzinfo is not None

    def test_default_branch_is_unknown(self):
        event = ChangeEvent(repo_path="/r", commit_hash="bbb", changed_files=["x.py"])
        assert event.branch == "unknown"


# ─── RepoStatus ───────────────────────────────────────────────────────────────

class TestRepoStatus:
    def test_to_dict_minimal(self):
        status = RepoStatus(repo_path="/repo")
        d = status.to_dict()
        assert d["repo_path"] == "/repo"
        assert d["watching"] is False
        assert d["last_commit"] is None
        assert d["last_checked"] is None

    def test_to_dict_with_timestamps(self):
        now = datetime.now(timezone.utc)
        status = RepoStatus(
            repo_path="/repo",
            watching=True,
            last_commit="abc123",
            last_checked=now,
            last_test_run=now,
            test_passing=True,
            change_count=3,
            debug_triggers=1,
        )
        d = status.to_dict()
        assert d["watching"] is True
        assert d["test_passing"] is True
        assert d["change_count"] == 3
        assert d["debug_triggers"] == 1
        assert "T" in d["last_checked"]  # ISO format

    def test_to_dict_recent_errors_capped_at_5(self):
        status = RepoStatus(repo_path="/r", errors=[str(i) for i in range(10)])
        d = status.to_dict()
        assert len(d["recent_errors"]) == 5


# ─── MonitorAgent properties ──────────────────────────────────────────────────

class TestMonitorAgentProperties:
    def test_name(self, monitor_agent):
        assert monitor_agent.name == "monitor"

    def test_description(self, monitor_agent):
        assert "monitor" in monitor_agent.description.lower()

    def test_system_prompt_content(self, monitor_agent):
        assert "root cause" in monitor_agent.system_prompt.lower()

    def test_tools_is_empty_list(self, monitor_agent):
        assert monitor_agent.tools == []


# ─── MonitorAgent git helpers ─────────────────────────────────────────────────

class TestMonitorAgentGitHelpers:
    def test_current_commit_valid_repo(self, sample_repo):
        result = MonitorAgent._current_commit(sample_repo)
        assert len(result) == 40  # full SHA

    def test_current_commit_invalid_path(self, tmp_path):
        result = MonitorAgent._current_commit(tmp_path)
        assert result == "unknown"

    def test_current_branch_valid_repo(self, sample_repo):
        result = MonitorAgent._current_branch(sample_repo)
        assert isinstance(result, str)
        assert result != ""

    def test_current_branch_invalid_path(self, tmp_path):
        result = MonitorAgent._current_branch(tmp_path)
        assert result == "unknown"

    def test_diff_files_same_ref(self, sample_repo):
        commit = MonitorAgent._current_commit(sample_repo)
        files = MonitorAgent._diff_files(sample_repo, commit, commit)
        assert files == []

    def test_diff_files_invalid_refs(self, sample_repo):
        files = MonitorAgent._diff_files(sample_repo, "nonexistent", "alsonotreal")
        assert files == []

    def test_diff_files_invalid_path(self, tmp_path):
        files = MonitorAgent._diff_files(tmp_path, "HEAD~1", "HEAD")
        assert files == []


# ─── MonitorAgent._run_tests ──────────────────────────────────────────────────

class TestMonitorAgentRunTests:
    @pytest.mark.asyncio
    async def test_run_tests_passes(self, monitor_agent, tmp_path):
        (tmp_path / "test_ok.py").write_text("def test_pass(): assert True")

        result = await monitor_agent._run_tests(tmp_path)
        assert result["passed"] is True
        assert "passed" in result["summary"].lower() or result["summary"]

    @pytest.mark.asyncio
    async def test_run_tests_fails(self, monitor_agent, tmp_path):
        (tmp_path / "test_fail.py").write_text("def test_fail(): assert False")

        result = await monitor_agent._run_tests(tmp_path)
        assert result["passed"] is False
        assert result["output"]

    @pytest.mark.asyncio
    async def test_run_tests_timeout(self, monitor_agent, tmp_path):
        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = await monitor_agent._run_tests(tmp_path)

        assert result["passed"] is False
        assert "timed out" in result["summary"].lower()
        assert result["output"] == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_run_tests_exception(self, monitor_agent, tmp_path):
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("no pytest")):
            result = await monitor_agent._run_tests(tmp_path)

        assert result["passed"] is False
        assert "no pytest" in result["summary"]


# ─── MonitorAgent.on_change_detected ─────────────────────────────────────────

class TestOnChangeDetected:
    @pytest.mark.asyncio
    async def test_updates_status(self, monitor_agent, sample_repo):
        status = RepoStatus(repo_path=str(sample_repo), watching=True)
        monitor_agent._repo_statuses[str(sample_repo)] = status

        event = ChangeEvent(
            repo_path=str(sample_repo),
            commit_hash="abc123",
            changed_files=["x.py"],
        )

        (sample_repo / "test_ok.py").write_text("def test_pass(): assert True")
        result = await monitor_agent.on_change_detected(sample_repo, event)

        assert "tests_passed" in result
        assert result["triggered_debug"] is False
        assert status.change_count == 1
        assert status.last_test_run is not None

    @pytest.mark.asyncio
    async def test_creates_status_if_missing(self, monitor_agent, sample_repo):
        event = ChangeEvent(
            repo_path=str(sample_repo),
            commit_hash="xyz",
            changed_files=[],
        )
        result = await monitor_agent.on_change_detected(sample_repo, event)
        assert "tests_passed" in result


# ─── MonitorAgent.on_test_failure ────────────────────────────────────────────

class TestOnTestFailure:
    @pytest.mark.asyncio
    async def test_returns_diagnosis(self, monitor_agent, sample_repo):
        status = RepoStatus(repo_path=str(sample_repo))
        monitor_agent._repo_statuses[str(sample_repo)] = status

        event = ChangeEvent(
            repo_path=str(sample_repo),
            commit_hash="deadbeef1234",
            changed_files=["app.py"],
        )
        test_results = {
            "passed": False,
            "summary": "1 failed",
            "output": "AssertionError: expected 1, got 2",
            "failed": 1,
        }

        result = await monitor_agent.on_test_failure(sample_repo, test_results, event)

        assert "diagnosis" in result
        assert result["commit"] == "deadbeef1234"
        assert "app.py" in result["changed_files"]
        assert status.debug_triggers == 1


# ─── MonitorAgent.generate_status_report ─────────────────────────────────────

class TestGenerateStatusReport:
    @pytest.mark.asyncio
    async def test_empty_report(self, monitor_agent):
        report = await monitor_agent.generate_status_report()
        assert report["total_repos"] == 0
        assert report["watching"] == 0
        assert "generated_at" in report

    @pytest.mark.asyncio
    async def test_report_with_repos(self, monitor_agent, sample_repo):
        monitor_agent._repo_statuses[str(sample_repo)] = RepoStatus(
            repo_path=str(sample_repo), watching=True
        )
        report = await monitor_agent.generate_status_report()
        assert report["total_repos"] == 1
        assert report["watching"] == 1


# ─── MonitorAgent.watch_repository ───────────────────────────────────────────

class TestWatchRepository:
    @pytest.mark.asyncio
    async def test_skips_non_git_dir(self, monitor_agent, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            await monitor_agent.watch_repository(str(tmp_path))
        assert str(tmp_path) not in monitor_agent._repo_statuses

    @pytest.mark.asyncio
    async def test_stop_watching_cancels_task(self, monitor_agent, sample_repo):
        task = asyncio.create_task(
            monitor_agent.watch_repository(str(sample_repo), poll_interval=9999)
        )
        monitor_agent._watch_tasks[str(sample_repo)] = task
        await asyncio.sleep(0.01)
        stopped = await monitor_agent.stop_watching(str(sample_repo))
        assert stopped is True
        assert task.done()

    @pytest.mark.asyncio
    async def test_stop_watching_nonexistent_returns_false(self, monitor_agent):
        result = await monitor_agent.stop_watching("/nonexistent/repo")
        assert result is False


# ─── ProactiveService ─────────────────────────────────────────────────────────

class TestProactiveService:
    @pytest.mark.asyncio
    async def test_initial_state(self):
        svc = ProactiveService()
        assert not svc.is_running

    @pytest.mark.asyncio
    async def test_get_status_when_no_agent(self):
        svc = ProactiveService()
        status = await svc.get_status()
        assert status["running"] is False
        assert status["watching"] == 0

    @pytest.mark.asyncio
    async def test_start_watching_creates_tasks(self, sample_repo):
        svc = ProactiveService()
        result = await svc.start_watching([str(sample_repo)], poll_interval=9999)
        assert str(sample_repo) in result["started"]
        assert result["total_watching"] == 1
        assert svc.is_running
        await svc.stop_watching()

    @pytest.mark.asyncio
    async def test_start_watching_skips_already_watched(self, sample_repo):
        svc = ProactiveService()
        await svc.start_watching([str(sample_repo)], poll_interval=9999)
        result = await svc.start_watching([str(sample_repo)], poll_interval=9999)
        assert str(sample_repo) in result["skipped"]
        await svc.stop_watching()

    @pytest.mark.asyncio
    async def test_stop_watching_all(self, sample_repo, tmp_path):
        import subprocess
        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        subprocess.run(["git", "init", str(repo2)], capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo2), capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo2), capture_output=True)
        (repo2 / "f.py").write_text("x=1")
        subprocess.run(["git", "add", "."], cwd=str(repo2), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo2), capture_output=True)

        svc = ProactiveService()
        await svc.start_watching([str(sample_repo), str(repo2)], poll_interval=9999)
        result = await svc.stop_watching()
        assert len(result["stopped"]) == 2
        assert not svc.is_running

    @pytest.mark.asyncio
    async def test_stop_specific_repo(self, sample_repo):
        svc = ProactiveService()
        await svc.start_watching([str(sample_repo)], poll_interval=9999)
        result = await svc.stop_watching([str(sample_repo)])
        assert str(sample_repo) in result["stopped"]

    @pytest.mark.asyncio
    async def test_stop_nonexistent_repo(self):
        svc = ProactiveService()
        result = await svc.stop_watching(["/not/watching/this"])
        assert "/not/watching/this" in result["not_found"]

    @pytest.mark.asyncio
    async def test_get_status_after_start(self, sample_repo):
        svc = ProactiveService()
        await svc.start_watching([str(sample_repo)], poll_interval=9999)
        status = await svc.get_status()
        assert status["running"] is True
        await svc.stop_watching()

    def test_get_proactive_service_singleton(self):
        svc1 = get_proactive_service()
        svc2 = get_proactive_service()
        assert svc1 is svc2


# ─── Monitor API endpoints ────────────────────────────────────────────────────

class TestMonitorEndpoints:
    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def mock_service(self):
        svc = MagicMock()
        svc.start_watching = AsyncMock(return_value={"started": ["/r"], "skipped": [], "total_watching": 1})
        svc.stop_watching = AsyncMock(return_value={"stopped": ["/r"], "not_found": [], "still_watching": 0})
        svc.get_status = AsyncMock(return_value={"running": True, "watching": 1, "total_repos": 1, "repos": {}, "generated_at": "now"})
        return svc

    def test_start_monitoring(self, client, mock_service):
        with patch("app.api.v1.endpoints.monitor.get_proactive_service", return_value=mock_service):
            resp = client.post(
                "/api/v1/monitor/start",
                json={"repo_paths": ["/repo"], "poll_interval": 60, "auto_debug": True},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "/r" in body["started"]

    def test_start_monitoring_error(self, client, mock_service):
        mock_service.start_watching = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("app.api.v1.endpoints.monitor.get_proactive_service", return_value=mock_service):
            resp = client.post(
                "/api/v1/monitor/start",
                json={"repo_paths": ["/repo"]},
            )
        assert resp.status_code == 500

    def test_stop_monitoring(self, client, mock_service):
        with patch("app.api.v1.endpoints.monitor.get_proactive_service", return_value=mock_service):
            resp = client.post("/api/v1/monitor/stop", json={"repo_paths": ["/r"]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_stop_monitoring_all(self, client, mock_service):
        with patch("app.api.v1.endpoints.monitor.get_proactive_service", return_value=mock_service):
            resp = client.post("/api/v1/monitor/stop", json={})
        assert resp.status_code == 200

    def test_stop_monitoring_error(self, client, mock_service):
        mock_service.stop_watching = AsyncMock(side_effect=RuntimeError("nope"))
        with patch("app.api.v1.endpoints.monitor.get_proactive_service", return_value=mock_service):
            resp = client.post("/api/v1/monitor/stop", json={})
        assert resp.status_code == 500

    def test_monitor_status(self, client, mock_service):
        with patch("app.api.v1.endpoints.monitor.get_proactive_service", return_value=mock_service):
            resp = client.get("/api/v1/monitor/status")
        assert resp.status_code == 200
        assert resp.json()["running"] is True

    def test_monitor_status_error(self, client, mock_service):
        mock_service.get_status = AsyncMock(side_effect=RuntimeError("fail"))
        with patch("app.api.v1.endpoints.monitor.get_proactive_service", return_value=mock_service):
            resp = client.get("/api/v1/monitor/status")
        assert resp.status_code == 500
