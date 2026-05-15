"""Tests for app.tools.code_analysis.security_scanner — SecurityScanner tool."""
import pytest
from app.tools.code_analysis.security_scanner import SecurityScanner


@pytest.fixture
def scanner():
    return SecurityScanner()


@pytest.fixture
def repo(tmp_path):
    safe = tmp_path / "safe.py"
    safe.write_text('x = 1\nprint(x)\n')
    vuln = tmp_path / "vuln.py"
    vuln.write_text('password = "super_secret_password"\neval(user_input)\n')
    return tmp_path


@pytest.mark.asyncio
async def test_missing_path_returns_error(scanner):
    result = await scanner.execute(path="")
    assert not result.success
    assert "required" in result.error


@pytest.mark.asyncio
async def test_nonexistent_path_returns_error(scanner):
    result = await scanner.execute(path="/nonexistent/path.py")
    assert not result.success
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_clean_file_returns_no_issues(tmp_path, scanner):
    clean = tmp_path / "clean.py"
    clean.write_text("def add(a, b):\n    return a + b\n")
    result = await scanner.execute(path=str(clean))
    assert result.success
    assert "No security issues" in result.output


@pytest.mark.asyncio
async def test_detects_hardcoded_password(tmp_path, scanner):
    f = tmp_path / "creds.py"
    f.write_text('db_password = "hunter2"\n')
    result = await scanner.execute(path=str(f))
    assert result.success
    assert "HIGH" in result.output


@pytest.mark.asyncio
async def test_detects_eval_usage(tmp_path, scanner):
    f = tmp_path / "risky.py"
    f.write_text('result = eval(user_input)\n')
    result = await scanner.execute(path=str(f))
    assert "HIGH" in result.output


@pytest.mark.asyncio
async def test_detects_exec_usage(tmp_path, scanner):
    f = tmp_path / "exec_file.py"
    f.write_text('exec(code_string)\n')
    result = await scanner.execute(path=str(f))
    assert "MEDIUM" in result.output


@pytest.mark.asyncio
async def test_scans_directory_recursively(repo, scanner):
    result = await scanner.execute(path=str(repo), recursive=True)
    assert result.success
    assert "finding" in result.output.lower()


@pytest.mark.asyncio
async def test_skips_venv_directory(tmp_path, scanner):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "secret.py").write_text('password = "do_not_scan_me"\n')
    (tmp_path / "main.py").write_text("x = 1\n")
    result = await scanner.execute(path=str(tmp_path))
    assert "do_not_scan_me" not in result.output


@pytest.mark.asyncio
async def test_name_property(scanner):
    assert scanner.name == "scan_security"


@pytest.mark.asyncio
async def test_description_property(scanner):
    assert "scan" in scanner.description.lower()


@pytest.mark.asyncio
async def test_shell_true_subprocess_detection(tmp_path, scanner):
    f = tmp_path / "shell_injection.py"
    f.write_text("import subprocess\nsubprocess.run(cmd, shell=True)\n")
    result = await scanner.execute(path=str(f))
    assert "HIGH" in result.output
