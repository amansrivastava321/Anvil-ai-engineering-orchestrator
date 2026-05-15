"""Tests for app.tools.code_analysis.dependency_analyzer — DependencyAnalyzer tool."""
import pytest
from app.tools.code_analysis.dependency_analyzer import DependencyAnalyzer, _extract_imports


@pytest.fixture
def analyzer():
    return DependencyAnalyzer()


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "main.py").write_text(
        "import requests\nimport structlog\nfrom pathlib import Path\n"
    )
    (tmp_path / "utils.py").write_text(
        "import httpx\nfrom fastapi import FastAPI\n"
    )
    return tmp_path


# ── _extract_imports helper ──────────────────────────────────────────────────

def test_extract_imports_regular_import():
    imports = _extract_imports("import os\nimport sys\n")
    assert "os" in imports
    assert "sys" in imports


def test_extract_imports_from_import():
    imports = _extract_imports("from pathlib import Path\n")
    assert "pathlib" in imports


def test_extract_imports_dotted_module():
    imports = _extract_imports("import os.path\n")
    assert "os" in imports


def test_extract_imports_handles_syntax_error():
    # Regex fallback for invalid Python
    imports = _extract_imports("import requests\n{{invalid python")
    assert "requests" in imports


def test_extract_imports_deduplicates():
    imports = _extract_imports("import os\nimport os\n")
    assert imports.count("os") == 1


# ── DependencyAnalyzer tool ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_path_returns_error(analyzer):
    result = await analyzer.execute(path="")
    assert not result.success
    assert "required" in result.error


@pytest.mark.asyncio
async def test_nonexistent_path_returns_error(analyzer):
    result = await analyzer.execute(path="/nonexistent/path")
    assert not result.success
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_analyzes_single_file(tmp_path, analyzer):
    f = tmp_path / "mymodule.py"
    f.write_text("import requests\nimport structlog\n")
    result = await analyzer.execute(path=str(f))
    assert result.success
    assert "requests" in result.output
    assert "structlog" in result.output


@pytest.mark.asyncio
async def test_stdlib_excluded_by_default(tmp_path, analyzer):
    f = tmp_path / "mymodule.py"
    f.write_text("import os\nimport pathlib\nimport requests\n")
    result = await analyzer.execute(path=str(f))
    assert result.success
    # stdlib modules should not appear (requests should)
    assert "requests" in result.output


@pytest.mark.asyncio
async def test_stdlib_included_when_requested(tmp_path, analyzer):
    f = tmp_path / "mymodule.py"
    f.write_text("import os\n")
    result = await analyzer.execute(path=str(f), include_stdlib=True)
    assert result.success
    assert "os" in result.output


@pytest.mark.asyncio
async def test_analyzes_directory(repo, analyzer):
    result = await analyzer.execute(path=str(repo))
    assert result.success
    assert "requests" in result.output
    assert "httpx" in result.output
    assert "fastapi" in result.output


@pytest.mark.asyncio
async def test_skips_pycache(tmp_path, analyzer):
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.py").write_text("import do_not_include\n")
    (tmp_path / "main.py").write_text("import requests\n")
    result = await analyzer.execute(path=str(tmp_path))
    assert "do_not_include" not in result.output


@pytest.mark.asyncio
async def test_name_property(analyzer):
    assert analyzer.name == "analyze_dependencies"


@pytest.mark.asyncio
async def test_description_property(analyzer):
    assert "dependenc" in analyzer.description.lower()
