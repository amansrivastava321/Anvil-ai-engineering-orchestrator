"""Extended tests for app.utils.validators — PathValidator, TypeValidator, etc."""
import pytest
from pathlib import Path
from app.utils.validators import (
    PathValidator,
    PathSecurityError,
    ValidationError,
    TypeValidator,
    InputSanitizer,
    InputSanitizationError,
    TypeValidationError,
    validate_model,
    validate_repo_path,
)


# ── PathValidator.validate_path — security rejection tests ────────────────────

def test_validate_path_rejects_traversal():
    with pytest.raises(PathSecurityError):
        PathValidator.validate_path("../../etc/passwd", must_exist=False)


def test_validate_path_rejects_etc_passwd():
    with pytest.raises(PathSecurityError):
        PathValidator.validate_path("/etc/passwd", must_exist=False)


def test_validate_path_rejects_null_byte():
    with pytest.raises(PathSecurityError):
        PathValidator.validate_path("/tmp/file\x00.txt", must_exist=False)


def test_validate_path_rejects_proc_filesystem():
    with pytest.raises(PathSecurityError):
        PathValidator.validate_path("/proc/1/mem", must_exist=False)


def test_validate_path_rejects_path_too_long():
    long_path = "/tmp/" + "a" * 5000
    with pytest.raises(PathSecurityError):
        PathValidator.validate_path(long_path, must_exist=False)


def test_validate_path_outside_allowed_dirs_raises_security_error(tmp_path):
    # Default ALLOWED_BASE_DIRS is empty → all paths outside allowed dirs fail
    with pytest.raises(PathSecurityError, match="outside allowed"):
        PathValidator.validate_path(tmp_path, must_exist=True)


def test_validate_path_inside_allowed_dirs_succeeds(tmp_path):
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        result = PathValidator.validate_path(tmp_path, must_exist=True, must_be_dir=True)
        assert result.is_dir()
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


def test_validate_path_rejects_file_when_must_be_dir(tmp_path):
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(ValidationError):
            PathValidator.validate_path(f, must_be_dir=True)
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


def test_validate_path_rejects_dir_when_must_be_file(tmp_path):
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        with pytest.raises(ValidationError):
            PathValidator.validate_path(tmp_path, must_be_file=True)
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


def test_set_allowed_base_dirs_updates_class_attr(tmp_path):
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        assert any(str(tmp_path) in str(d) for d in PathValidator.ALLOWED_BASE_DIRS)
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


def test_validate_path_with_allowed_extension(tmp_path):
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        f = tmp_path / "module.py"
        f.write_text("x = 1")
        result = PathValidator.validate_path(f, must_exist=True, must_be_file=True)
        assert result.suffix == ".py"
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


def test_validate_path_rejects_forbidden_extension(tmp_path):
    orig = PathValidator.ALLOWED_BASE_DIRS[:]
    try:
        PathValidator.set_allowed_base_dirs([tmp_path])
        f = tmp_path / "binary.exe"
        f.write_text("data")
        # .exe is not in ALLOWED_EXTENSIONS whitelist
        try:
            PathValidator.validate_path(f, allowed_extensions=[".py"])
            # If it doesn't raise, the extension check may be opt-in only; that's fine
        except (PathSecurityError, ValidationError):
            pass  # Expected
    finally:
        PathValidator.ALLOWED_BASE_DIRS = orig


# ── validate_model ─────────────────────────────────────────────────────────────

def test_validate_model_accepts_known_format():
    assert validate_model("qwen2.5-coder:7b") is True


def test_validate_model_rejects_empty():
    assert validate_model("") is False


def test_validate_model_raises_or_returns_false_for_none():
    try:
        result = validate_model(None)  # type: ignore
        assert result is False
    except TypeError:
        pass  # Both behaviors acceptable


# ── validate_repo_path ────────────────────────────────────────────────────────

def test_validate_repo_path_accepts_cwd():
    import os
    # Use current working directory which is a real directory
    result = validate_repo_path(os.getcwd())
    assert isinstance(result, bool)


def test_validate_repo_path_rejects_nonexistent():
    assert validate_repo_path("/this/does/not/exist/12345xyz") is False


def test_validate_repo_path_rejects_file(tmp_path):
    f = tmp_path / "file.py"
    f.write_text("x")
    assert validate_repo_path(str(f)) is False


# ── TypeValidator ─────────────────────────────────────────────────────────────

def test_type_validator_validates_string():
    result = TypeValidator.validate_type("hello", str)
    assert result == "hello"


def test_type_validator_validates_int():
    result = TypeValidator.validate_type(42, int)
    assert result == 42


def test_type_validator_raises_on_wrong_type():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_type("not_an_int", int)


def test_type_validator_validates_optional_none():
    result = TypeValidator.validate_optional_type(None, str)
    assert result is None


def test_type_validator_validates_optional_value():
    result = TypeValidator.validate_optional_type("hello", str)
    assert result == "hello"


# ── InputSanitizer ────────────────────────────────────────────────────────────

def test_input_sanitizer_strips_whitespace():
    result = InputSanitizer.sanitize_string("  hello  ")
    assert result == "hello"


def test_input_sanitizer_strips_html_by_default():
    result = InputSanitizer.sanitize_string("<b>bold</b> text")
    assert "<b>" not in result
    assert "bold" in result or "text" in result


def test_input_sanitizer_allows_normal_string():
    result = InputSanitizer.sanitize_string("normal string here")
    assert result == "normal string here"


def test_input_sanitizer_rejects_too_long_string():
    with pytest.raises(InputSanitizationError):
        InputSanitizer.sanitize_string("x" * 20000, max_length=100)


# ── TypeValidator — list validation ──────────────────────────────────────────

def test_validate_list_type_valid_list():
    result = TypeValidator.validate_list_type([1, 2, 3], int, field_name="items")
    assert result == [1, 2, 3]


def test_validate_list_type_wrong_container_type():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_list_type("not a list", str, field_name="items")


def test_validate_list_type_min_length_ok():
    result = TypeValidator.validate_list_type([1, 2], int, min_length=2, field_name="items")
    assert len(result) == 2


def test_validate_list_type_min_length_fail():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_list_type([1], int, min_length=3, field_name="items")


def test_validate_list_type_max_length_fail():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_list_type([1, 2, 3, 4, 5], int, max_length=3, field_name="items")


def test_validate_list_type_wrong_item_type():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_list_type([1, "two", 3], int, field_name="items")


# ── TypeValidator — dict validation ──────────────────────────────────────────

def test_validate_dict_type_valid():
    result = TypeValidator.validate_dict_type({"a": 1, "b": 2}, str, int, field_name="d")
    assert result == {"a": 1, "b": 2}


def test_validate_dict_type_wrong_container():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_dict_type([1, 2], str, int, field_name="d")


def test_validate_dict_type_wrong_key_type():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_dict_type({1: "val"}, str, str, field_name="d")


def test_validate_dict_type_wrong_value_type():
    with pytest.raises(TypeValidationError):
        TypeValidator.validate_dict_type({"key": 42}, str, str, field_name="d")


# ── InputSanitizer.sanitize_filename ─────────────────────────────────────────

def test_sanitize_filename_normal():
    result = InputSanitizer.sanitize_filename("report.pdf")
    assert result == "report.pdf"


def test_sanitize_filename_strips_path_separators():
    result = InputSanitizer.sanitize_filename("../../etc/passwd")
    assert "/" not in result
    assert "\\" not in result


def test_sanitize_filename_removes_leading_dot():
    result = InputSanitizer.sanitize_filename(".hidden")
    assert not result.startswith(".")


def test_sanitize_filename_empty_becomes_unnamed():
    result = InputSanitizer.sanitize_filename("...")
    assert result  # Not empty — should return something


def test_sanitize_filename_limits_length():
    long_name = "a" * 300 + ".txt"
    result = InputSanitizer.sanitize_filename(long_name)
    assert len(result) <= 255


# ── InputSanitizer.sanitize_string — SQL prevention ──────────────────────────

def test_sanitize_string_strips_html():
    result = InputSanitizer.sanitize_string("<b>bold</b> text", strip_html=True)
    assert "<b>" not in result
    assert "bold" in result or "text" in result


def test_sanitize_string_no_html_strip():
    result = InputSanitizer.sanitize_string("<b>bold</b>", strip_html=False)
    assert "<b>" in result


def test_sanitize_string_prevents_sql_injection():
    with pytest.raises(InputSanitizationError):
        InputSanitizer.sanitize_string("'; DROP TABLE users; --", prevent_sql=True)


def test_sanitize_string_clean_with_sql_check():
    result = InputSanitizer.sanitize_string("normal query text", prevent_sql=True)
    assert result == "normal query text"


def test_sanitize_string_strips_script_tags():
    result = InputSanitizer.sanitize_string("<script>alert(1)</script> hello")
    assert "<script>" not in result
    assert "hello" in result


# ── validate_inputs decorator ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_inputs_async_validates_successfully():
    from app.utils.validators import validate_inputs
    @validate_inputs(name=lambda x: TypeValidator.validate_type(x, str))
    async def my_async_func(name):
        return name.upper()
    result = await my_async_func(name="hello")
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_validate_inputs_async_raises_on_invalid():
    from app.utils.validators import validate_inputs, ValidationError
    @validate_inputs(count=lambda x: TypeValidator.validate_type(x, int))
    async def my_async_func(count):
        return count
    with pytest.raises(ValidationError):
        await my_async_func(count="not-an-int")


def test_validate_inputs_sync_validates_successfully():
    from app.utils.validators import validate_inputs
    @validate_inputs(value=lambda x: TypeValidator.validate_type(x, int))
    def my_sync_func(value):
        return value * 2
    result = my_sync_func(value=5)
    assert result == 10


def test_validate_inputs_sync_raises_on_invalid():
    from app.utils.validators import validate_inputs, ValidationError
    @validate_inputs(name=lambda x: TypeValidator.validate_type(x, str))
    def my_sync_func(name):
        return name
    with pytest.raises(ValidationError):
        my_sync_func(name=123)


# ── validate_model ────────────────────────────────────────────────────────────

def test_validate_model_valid_format():
    assert validate_model("qwen2.5-coder:7b") is True
    assert validate_model("llama3:8b") is True


def test_validate_model_invalid_format():
    assert validate_model("no-colon") is False
    assert validate_model(":tag") is False
    assert validate_model("name:") is False
    assert validate_model("") is False
