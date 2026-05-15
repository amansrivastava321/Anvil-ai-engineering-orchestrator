# tests/unit/test_tools.py
import pytest
from pathlib import Path
from app.tools.file_system.file_reader import FileReader, ToolResult
from app.tools.file_system.file_writer import FileWriter
from app.tools.testing.test_runner import TestRunner


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "hello.py").write_text("print('hello')")
    return tmp_path


# ── ToolResult ────────────────────────────────────────────────────────────────

def test_tool_result_success():
    r = ToolResult("read_file", True, "content")
    assert r.success
    assert r.output == "content"
    assert r.error is None


def test_tool_result_failure():
    r = ToolResult("read_file", False, "", error="not found")
    assert not r.success
    assert r.error == "not found"


# ── FileReader ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_file_reader_reads_existing_file(repo):
    reader = FileReader(str(repo))
    result = await reader.execute("src/hello.py")
    assert result.success
    assert "print" in result.output


@pytest.mark.asyncio
async def test_file_reader_returns_failure_for_missing_file(repo):
    reader = FileReader(str(repo))
    result = await reader.execute("nonexistent.py")
    assert not result.success
    assert result.error is not None
    assert result.output == ""


@pytest.mark.asyncio
async def test_file_reader_duration_recorded(repo):
    reader = FileReader(str(repo))
    result = await reader.execute("src/hello.py")
    assert result.duration_ms >= 0


# ── FileWriter ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_file_writer_writes_new_file(repo):
    writer = FileWriter(str(repo))
    result = await writer.execute("src/new_file.py", "x = 1")
    assert result.success
    assert (repo / "src" / "new_file.py").read_text() == "x = 1"


@pytest.mark.asyncio
async def test_file_writer_overwrites_existing_file(repo):
    writer = FileWriter(str(repo))
    await writer.execute("src/hello.py", "x = 2")
    assert (repo / "src" / "hello.py").read_text() == "x = 2"


@pytest.mark.asyncio
async def test_file_writer_fails_for_nonexistent_parent_dir(repo):
    writer = FileWriter(str(repo))
    result = await writer.execute("nonexistent_dir/file.py", "x = 1")
    assert not result.success
    assert result.error is not None


# ── TestRunner ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_runs_passing_tests(tmp_path):
    (tmp_path / "test_sample.py").write_text(
        "def test_always_passes():\n    assert 1 == 1\n"
    )
    runner = TestRunner(str(tmp_path))
    result = await runner.execute(test_path="test_sample.py")
    assert result.success
    assert "passed" in result.output


@pytest.mark.asyncio
async def test_runner_reports_failing_tests(tmp_path):
    (tmp_path / "test_fail.py").write_text(
        "def test_always_fails():\n    assert 1 == 2\n"
    )
    runner = TestRunner(str(tmp_path))
    result = await runner.execute(test_path="test_fail.py")
    assert not result.success
    assert result.error is not None


@pytest.mark.asyncio
async def test_runner_with_extra_args(tmp_path):
    (tmp_path / "test_extra.py").write_text(
        "def test_ok():\n    assert True\n"
    )
    runner = TestRunner(str(tmp_path))
    result = await runner.execute(test_path="test_extra.py", extra_args="-q")
    assert isinstance(result.success, bool)


@pytest.mark.asyncio
async def test_runner_timeout_returns_failure():
    from unittest.mock import AsyncMock, patch, MagicMock
    import asyncio
    runner = TestRunner("/tmp")
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
         patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
        result = await runner.execute()
    assert not result.success
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_runner_subprocess_exception_returns_failure():
    from unittest.mock import patch
    runner = TestRunner("/tmp")
    with patch("asyncio.create_subprocess_exec", side_effect=OSError("no such file")):
        result = await runner.execute()
    assert not result.success
    assert result.error is not None


def test_runner_name_and_description():
    runner = TestRunner("/tmp")
    assert runner.name == "run_tests"
    assert "pytest" in runner.description.lower() or "test" in runner.description.lower()
