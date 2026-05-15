"""Tests for app.tools.file_system.file_search — FileSearch tool."""
import pytest
from app.tools.file_system.file_search import FileSearch


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello(): return 'hello world'\n")
    (tmp_path / "src" / "utils.py").write_text("import os\ndef get_path(): return os.getcwd()\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("from src.main import hello\n")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "secret.py").write_text("password = 'should_not_appear'")
    return tmp_path


@pytest.fixture
def searcher():
    return FileSearch()


@pytest.mark.asyncio
async def test_missing_repo_path_returns_error(searcher):
    result = await searcher.execute(repo_path="", query="hello")
    assert not result.success
    assert "required" in result.error


@pytest.mark.asyncio
async def test_missing_query_returns_error(searcher, repo):
    result = await searcher.execute(repo_path=str(repo), query="")
    assert not result.success
    assert "required" in result.error


@pytest.mark.asyncio
async def test_nonexistent_repo_returns_error(searcher):
    result = await searcher.execute(repo_path="/nonexistent/path", query="hello")
    assert not result.success
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_filename_match(searcher, repo):
    result = await searcher.execute(repo_path=str(repo), query="main")
    assert result.success
    assert "main.py" in result.output


@pytest.mark.asyncio
async def test_content_match(searcher, repo):
    result = await searcher.execute(repo_path=str(repo), query="hello world")
    assert result.success
    assert "main.py" in result.output
    assert "hello world" in result.output


@pytest.mark.asyncio
async def test_no_results_returns_friendly_message(searcher, repo):
    result = await searcher.execute(repo_path=str(repo), query="XYZNOTHERE")
    assert result.success
    assert "No results" in result.output


@pytest.mark.asyncio
async def test_skips_git_directory(searcher, repo):
    result = await searcher.execute(repo_path=str(repo), query="should_not_appear")
    # The .git/secret.py file should not be found — result should be "No results"
    assert "No results" in result.output or result.output.count("should_not_appear") == 1


@pytest.mark.asyncio
async def test_max_results_limit(searcher, repo):
    for i in range(30):
        (repo / f"file_{i}.py").write_text("target_token = True\n")
    result = await searcher.execute(repo_path=str(repo), query="target_token", max_results=5)
    assert result.success
    count = result.output.count("target_token")
    assert count <= 6  # header line + 5 results


@pytest.mark.asyncio
async def test_name_property(searcher):
    assert searcher.name == "search_files"


@pytest.mark.asyncio
async def test_description_property(searcher):
    assert "search" in searcher.description.lower()
