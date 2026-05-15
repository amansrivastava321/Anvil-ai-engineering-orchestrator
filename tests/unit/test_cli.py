"""Tests for the ae CLI tool."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from cli.ae import cli, detect_workflow, _format_duration, main


# ── detect_workflow ─────────────────────────────────────────────────────────


class TestDetectWorkflow:
    def test_debug_keywords(self):
        for kw in ["fix", "bug", "error", "debug", "crash", "broken", "fail"]:
            assert detect_workflow(f"{kw} this") == "debug_analysis"

    def test_architecture_keywords(self):
        for kw in ["architecture", "structure", "design", "overview"]:
            assert detect_workflow(kw) == "architecture_analysis"

    def test_test_keywords(self):
        for kw in ["test", "tests", "coverage", "pytest"]:
            assert detect_workflow(f"{kw} this") == "test_generation"

    def test_refactor_keywords(self):
        for kw in ["refactor", "clean", "improve", "simplify"]:
            assert detect_workflow(kw) == "code_refactoring"

    def test_doc_keywords(self):
        for kw in ["document", "explain", "readme", "docs"]:
            assert detect_workflow(kw) == "documentation"

    def test_generate_keywords(self):
        for kw in ["write", "create", "generate", "implement", "build"]:
            assert detect_workflow(kw) == "code_generation"

    def test_review_keywords(self):
        for kw in ["review", "audit", "inspect"]:
            assert detect_workflow(kw) == "code_review"

    def test_fallback_to_general_qa(self):
        assert detect_workflow("what is going on here?") == "general_qa"

    def test_case_insensitive(self):
        assert detect_workflow("FIX the BUG") == "debug_analysis"

    def test_whole_word_matching(self):
        # "inspect" must not match keywords from other workflows as substrings
        assert detect_workflow("inspect this module") == "code_review"

    def test_first_match_wins(self):
        assert detect_workflow("fix and review") == "debug_analysis"


# ── format_duration ─────────────────────────────────────────────────────────


class TestFormatDuration:
    def test_under_second(self):
        assert _format_duration(250) == "250ms"

    def test_over_second(self):
        assert _format_duration(2500) == "2.5s"

    def test_exactly_one_second(self):
        assert _format_duration(1000) == "1.0s"


# ── CLI runner helpers ───────────────────────────────────────────────────────


@pytest.fixture()
def runner():
    return CliRunner()


def _mock_response(data: Any, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    return resp


def _orch_response(**kwargs) -> dict:
    base = {
        "execution_id": "abc123",
        "status": "completed",
        "workflow_type": "general_qa",
        "model_used": "deepseek-r1:7b",
        "response": "This is the answer.",
        "artifacts": [],
        "graphify_available": False,
        "files_analyzed": [],
        "tokens_used": 100,
        "duration_ms": 1234.0,
        "warnings": [],
        "error": None,
    }
    base.update(kwargs)
    return base


def _make_client_ctx(mc: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mc)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _fake_thread_factory(fn=None):
    """Thread whose start() calls target() synchronously."""
    def factory(target=None, daemon=None):
        t = MagicMock()
        t.is_alive.return_value = False
        t.start = lambda: target()
        t.join = MagicMock()
        return t
    return factory


# ── status command ───────────────────────────────────────────────────────────


class TestStatusCommand:
    def test_shows_system_health(self, runner, tmp_path):
        health = {
            "ollama": {"status": "healthy", "model_count": 3},
            "graphify": {"available": True, "repos_analyzed": 2},
            "skills": {"loaded": True, "count": 5},
        }
        monitor = {"watching": [str(tmp_path)]}
        evolution = {"total_cycles_run": 4, "total_improvements_applied": 7}
        stats = {
            "total_executions": 50, "successful_executions": 45,
            "active_executions": 1, "avg_duration_ms": 800,
        }

        mc = MagicMock()
        mc.get.side_effect = lambda url: _mock_response(
            health if "system/status" in url
            else stats if "agent/stats" in url
            else monitor if "monitor/status" in url
            else evolution
        )

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "Ollama" in result.output
        assert "Evolution" in result.output

    def test_zero_executions_dash(self, runner):
        """Success rate shows — when total_executions is 0."""
        mc = MagicMock()
        mc.get.side_effect = lambda url: _mock_response(
            {"ollama": {}, "graphify": {}, "skills": {}} if "system/status" in url
            else {"total_executions": 0, "successful_executions": 0, "active_executions": 0} if "agent/stats" in url
            else {"watching": []} if "monitor/status" in url
            else {"total_cycles_run": 0, "total_improvements_applied": 0}
        )
        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0, result.output
        assert "—" in result.output

    def test_connection_error_exits_1(self, runner):
        mc = MagicMock()
        mc.get.side_effect = httpx.ConnectError("refused")

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 1
        assert "Cannot connect" in result.output


# ── models command ───────────────────────────────────────────────────────────


class TestModelsCommand:
    def test_lists_models(self, runner):
        models_data = {
            "models": [
                {"name": "deepseek-r1:7b", "tier": "balanced", "size": "4.7GB",
                 "task_types": ["code_generation"], "weight": 1.0},
                {"name": "phi4-mini:latest", "tier": "fast", "size": "2.5GB",
                 "task_types": ["general_qa"], "weight": 0.8},
            ],
            "total": 2,
        }
        mc = MagicMock()
        mc.get.return_value = _mock_response(models_data)

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0, result.output
        assert "deepseek-r1:7b" in result.output
        assert "phi4-mini" in result.output

    def test_empty_model_list_warning(self, runner):
        mc = MagicMock()
        mc.get.return_value = _mock_response({"models": [], "total": 0})

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0, result.output
        assert "No models" in result.output

    def test_connection_error(self, runner):
        mc = MagicMock()
        mc.get.side_effect = httpx.ConnectError("refused")

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 1


# ── watch command ────────────────────────────────────────────────────────────


class TestWatchCommand:
    def test_starts_monitoring(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response({"status": "ok"})

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["watch", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "watching" in result.output.lower()

    def test_invalid_repo_exits(self, runner):
        result = runner.invoke(cli, ["watch", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_server_error_exits(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response({"error": "boom"}, status=500)

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["watch", str(tmp_path)])

        assert result.exit_code == 1

    def test_default_repo_is_cwd(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response({"status": "ok"})

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.Path.cwd", return_value=tmp_path):
            result = runner.invoke(cli, ["watch"])

        assert result.exit_code == 0, result.output

    def test_connection_error(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.side_effect = httpx.ConnectError("refused")

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["watch", str(tmp_path)])

        assert result.exit_code == 1
        assert "Cannot connect" in result.output


# ── dashboard command ────────────────────────────────────────────────────────


class TestDashboardCommand:
    def test_opens_browser(self, runner):
        with patch("webbrowser.open") as mock_open:
            result = runner.invoke(cli, ["dashboard"])
        assert result.exit_code == 0, result.output
        mock_open.assert_called_once_with("http://localhost:8008/dashboard")

    def test_custom_port(self, runner):
        with patch("webbrowser.open") as mock_open:
            result = runner.invoke(cli, ["dashboard", "--port", "9000"])
        assert result.exit_code == 0, result.output
        mock_open.assert_called_once_with("http://localhost:9000/dashboard")


# ── run subcommand (prompt execution) ───────────────────────────────────────


class TestRunCommand:
    def test_help_shown_with_no_args(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_invalid_repo_exits(self, runner):
        result = runner.invoke(cli, ["run", "--repo", "/nonexistent", "hello"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_auto_detects_workflow_and_runs(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response(_orch_response(workflow_type="debug_analysis"))

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "fix the bug"])

        assert result.exit_code == 0, result.output

    def test_server_error_displayed(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response({"error": "model offline"}, status=500)

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "fix the bug"])

        assert result.exit_code == 1

    def test_connection_error_on_prompt(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.side_effect = httpx.ConnectError("refused")

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "fix the bug"])

        assert result.exit_code == 1
        assert "Cannot connect" in result.output

    def test_stream_flag(self, runner, tmp_path):
        mc = MagicMock()
        stream_resp = MagicMock()
        stream_resp.status_code = 200
        stream_resp.__enter__ = MagicMock(return_value=stream_resp)
        stream_resp.__exit__ = MagicMock(return_value=False)
        stream_resp.iter_text.return_value = iter(["Hello ", "world"])
        mc.stream.return_value = stream_resp

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "--stream", "write tests"])

        assert result.exit_code == 0, result.output

    def test_stream_connection_error(self, runner, tmp_path):
        mc = MagicMock()
        mc.stream.side_effect = httpx.ConnectError("refused")

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "--stream", "write tests"])

        assert result.exit_code == 1
        assert "Cannot connect" in result.output

    def test_preferred_model_passed(self, runner, tmp_path):
        captured: dict = {}
        mc = MagicMock()

        def capture_post(url, json=None):
            captured.update(json or {})
            return _mock_response(_orch_response(model_used="phi4-mini:latest"))

        mc.post.side_effect = capture_post

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            runner.invoke(cli, ["run", "--repo", str(tmp_path), "--model", "phi4-mini:latest", "review"])

        assert captured.get("preferred_model") == "phi4-mini:latest"

    def test_graphify_shown_when_available(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response(
            _orch_response(graphify_available=True, files_analyzed=["a.py", "b.py"])
        )

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "explain this"])

        assert result.exit_code == 0, result.output
        assert "loaded" in result.output or "2" in result.output

    def test_timeout_error(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.side_effect = httpx.TimeoutException("timed out")

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "explain this"])

        assert result.exit_code == 1
        assert "timed out" in result.output.lower()

    def test_stream_error_status(self, runner, tmp_path):
        mc = MagicMock()
        stream_resp = MagicMock()
        stream_resp.status_code = 500
        stream_resp.__enter__ = MagicMock(return_value=stream_resp)
        stream_resp.__exit__ = MagicMock(return_value=False)
        mc.stream.return_value = stream_resp

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "--stream", "write tests"])

        assert result.exit_code == 1

    def test_response_with_warnings(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response(
            _orch_response(warnings=["Model fallback used"])
        )

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "explain this"])

        assert result.exit_code == 0, result.output
        assert "fallback" in result.output.lower()

    def test_failed_response_shows_error(self, runner, tmp_path):
        mc = MagicMock()
        mc.post.return_value = _mock_response(
            _orch_response(status="failed", error="Agent crashed")
        )

        with patch("cli.ae._client", return_value=_make_client_ctx(mc)), \
             patch("cli.ae.threading.Thread", side_effect=_fake_thread_factory()):
            result = runner.invoke(cli, ["run", "--repo", str(tmp_path), "explain this"])

        assert result.exit_code == 0  # error is rendered, not sys.exit
        assert "Agent crashed" in result.output


# ── main() entry point routing ───────────────────────────────────────────────


class TestMainRouting:
    def test_inserts_run_for_prompt(self):
        """main() inserts 'run' when first positional is not a subcommand."""
        import sys as _sys
        original = _sys.argv[:]
        _sys.argv = ["ae", "fix the bug"]
        try:
            with patch("cli.ae.cli") as mock_cli:
                main()
            assert _sys.argv[1] == "run"
        finally:
            _sys.argv = original

    def test_does_not_insert_run_for_subcommand(self):
        import sys as _sys
        original = _sys.argv[:]
        _sys.argv = ["ae", "status"]
        try:
            with patch("cli.ae.cli") as mock_cli:
                main()
            assert _sys.argv[1] == "status"
        finally:
            _sys.argv = original

    def test_does_not_insert_run_for_option_only(self):
        import sys as _sys
        original = _sys.argv[:]
        _sys.argv = ["ae", "--help"]
        try:
            with patch("cli.ae.cli") as mock_cli:
                main()
            assert _sys.argv[1] == "--help"
        finally:
            _sys.argv = original
