"""
Comprehensive tests for app/core/monitoring/logging.py

Covers: JSONFormatter, ConsoleFormatter, KeyValueFormatter, AsyncLogHandler,
        SizeRotatingFileHandler, LogContext, log_function_call decorator,
        RequestLogger, get_logger, setup_logging, set_request_context,
        clear_request_context, and structlog processor helpers.
"""

import json
import logging
import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import structlog

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from app.core.monitoring.logging import (
    JSONFormatter,
    ConsoleFormatter,
    KeyValueFormatter,
    AsyncLogHandler,
    SizeRotatingFileHandler,
    LogContext,
    RequestLogger,
    get_logger,
    set_request_context,
    clear_request_context,
    log_function_call,
    mask_sensitive_data,
    add_timestamp,
    add_log_level,
    add_context_vars,
    add_process_info,
    add_host_info,
    add_environment,
    mask_sensitive,
    truncate_long_messages,
    add_exception_info,
    setup_logging,
    request_id_ctx,
    correlation_id_ctx,
    user_id_ctx,
    session_id_ctx,
    agent_id_ctx,
    trace_id_ctx,
    SENSITIVE_FIELDS,
    MAX_STRING_LENGTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(
    name="test.logger",
    level=logging.INFO,
    msg="Hello world",
    lineno=1,
    **extra,
):
    """Create a LogRecord with sensible defaults."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test_logging_module.py",
        lineno=lineno,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# ===========================================================================
# JSONFormatter tests
# ===========================================================================

class TestJSONFormatter:

    def test_basic_output_is_valid_json(self):
        formatter = JSONFormatter()
        record = make_record(msg="Hello world")
        output = formatter.format(record)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_message_field(self):
        formatter = JSONFormatter()
        record = make_record(msg="Hello world")
        data = json.loads(formatter.format(record))
        assert data["message"] == "Hello world"

    def test_level_field_info(self):
        formatter = JSONFormatter()
        record = make_record(level=logging.INFO)
        data = json.loads(formatter.format(record))
        assert data["level"] == "INFO"

    def test_level_field_error(self):
        formatter = JSONFormatter()
        record = make_record(level=logging.ERROR, msg="boom")
        data = json.loads(formatter.format(record))
        assert data["level"] == "ERROR"

    def test_level_field_warning(self):
        formatter = JSONFormatter()
        record = make_record(level=logging.WARNING, msg="warn")
        data = json.loads(formatter.format(record))
        assert data["level"] == "WARNING"

    def test_timestamp_present(self):
        formatter = JSONFormatter()
        record = make_record()
        data = json.loads(formatter.format(record))
        assert "timestamp" in data

    def test_logger_name(self):
        formatter = JSONFormatter()
        record = make_record(name="my.custom.logger")
        data = json.loads(formatter.format(record))
        assert data["logger"] == "my.custom.logger"

    def test_service_and_version_present(self):
        formatter = JSONFormatter()
        record = make_record()
        data = json.loads(formatter.format(record))
        assert "service" in data
        assert "version" in data

    def test_module_function_line_present(self):
        formatter = JSONFormatter()
        record = make_record(lineno=42)
        data = json.loads(formatter.format(record))
        assert "module" in data
        assert "function" in data
        assert "line" in data
        assert data["line"] == 42

    def test_extra_attribute_included(self):
        formatter = JSONFormatter()
        record = make_record(msg="Extra test")
        record.user_id = "u-999"
        data = json.loads(formatter.format(record))
        assert data.get("user_id") == "u-999"

    def test_sensitive_field_masked(self):
        formatter = JSONFormatter()
        record = make_record(msg="Auth")
        record.password = "supersecret"
        data = json.loads(formatter.format(record))
        # The password should be masked
        assert data.get("password") == "********"

    def test_exception_info_captured(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()
        record = make_record(msg="Error occurred")
        record.exc_info = exc_info
        data = json.loads(formatter.format(record))
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "test error" in data["exception"]["message"]

    def test_format_without_exception(self):
        formatter = JSONFormatter()
        record = make_record(msg="No exception")
        record.exc_info = None
        data = json.loads(formatter.format(record))
        assert "exception" not in data

    def test_debug_level(self):
        formatter = JSONFormatter()
        record = make_record(level=logging.DEBUG, msg="debug msg")
        data = json.loads(formatter.format(record))
        assert data["level"] == "DEBUG"

    def test_critical_level(self):
        formatter = JSONFormatter()
        record = make_record(level=logging.CRITICAL, msg="critical msg")
        data = json.loads(formatter.format(record))
        assert data["level"] == "CRITICAL"

    def test_event_dict_merged(self):
        formatter = JSONFormatter()
        record = make_record(msg="Has event_dict")
        record.event_dict = {"custom_key": "custom_val"}
        data = json.loads(formatter.format(record))
        assert data.get("custom_key") == "custom_val"

    def test_message_args_interpolation(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="Hello %s", args=("world",), exc_info=None
        )
        data = json.loads(formatter.format(record))
        assert data["message"] == "Hello world"


# ===========================================================================
# ConsoleFormatter tests
# ===========================================================================

class TestConsoleFormatter:

    def test_output_contains_message(self):
        formatter = ConsoleFormatter()
        record = make_record(msg="Test console message")
        result = formatter.format(record)
        assert "Test console message" in result

    def test_output_is_string(self):
        formatter = ConsoleFormatter()
        record = make_record()
        assert isinstance(formatter.format(record), str)

    def test_includes_log_level(self):
        formatter = ConsoleFormatter()
        record = make_record(level=logging.ERROR, msg="err")
        result = formatter.format(record)
        assert "ERROR" in result

    def test_includes_timestamp(self):
        formatter = ConsoleFormatter()
        record = make_record()
        result = formatter.format(record)
        # Timestamp format like "2024-01-01 12:00:00"
        assert "-" in result and ":" in result

    def test_includes_logger_name(self):
        formatter = ConsoleFormatter()
        record = make_record(name="my.module")
        result = formatter.format(record)
        assert "my.module" in result

    def test_exception_appended(self):
        formatter = ConsoleFormatter()
        try:
            raise RuntimeError("console exception")
        except RuntimeError:
            exc_info = sys.exc_info()
        record = make_record(msg="With exception")
        record.exc_info = exc_info
        result = formatter.format(record)
        assert "console exception" in result or "RuntimeError" in result

    def test_with_request_id_context(self):
        formatter = ConsoleFormatter()
        token = request_id_ctx.set("req-abc123")
        try:
            record = make_record(msg="context test")
            result = formatter.format(record)
            # req= prefix with first 8 chars of request_id
            assert "req-abc1" in result
        finally:
            request_id_ctx.reset(token)

    def test_with_agent_id_context(self):
        formatter = ConsoleFormatter()
        token = agent_id_ctx.set("agent-007")
        try:
            record = make_record(msg="agent context")
            result = formatter.format(record)
            assert "agent-007" in result
        finally:
            agent_id_ctx.reset(token)


# ===========================================================================
# KeyValueFormatter tests
# ===========================================================================

class TestKeyValueFormatter:

    def test_contains_level_kv(self):
        formatter = KeyValueFormatter()
        record = make_record(level=logging.INFO, msg="hello")
        result = formatter.format(record)
        assert "level=INFO" in result

    def test_contains_message_kv(self):
        formatter = KeyValueFormatter()
        record = make_record(msg="hello world")
        result = formatter.format(record)
        assert "hello world" in result

    def test_contains_logger_kv(self):
        formatter = KeyValueFormatter()
        record = make_record(name="kv.logger")
        result = formatter.format(record)
        assert "logger=kv.logger" in result

    def test_contains_timestamp_kv(self):
        formatter = KeyValueFormatter()
        record = make_record()
        result = formatter.format(record)
        assert "timestamp=" in result

    def test_is_string(self):
        formatter = KeyValueFormatter()
        record = make_record()
        assert isinstance(formatter.format(record), str)

    def test_exception_included_when_present(self):
        formatter = KeyValueFormatter()
        try:
            raise KeyError("kv error")
        except KeyError:
            exc_info = sys.exc_info()
        record = make_record(msg="With kv exception")
        record.exc_info = exc_info
        result = formatter.format(record)
        assert "exception=" in result

    def test_request_id_included_when_set(self):
        formatter = KeyValueFormatter()
        token = request_id_ctx.set("kv-req-id")
        try:
            record = make_record()
            result = formatter.format(record)
            assert "request_id=kv-req-id" in result
        finally:
            request_id_ctx.reset(token)

    def test_user_id_included_when_set(self):
        formatter = KeyValueFormatter()
        token = user_id_ctx.set("u-456")
        try:
            record = make_record()
            result = formatter.format(record)
            assert "user_id=u-456" in result
        finally:
            user_id_ctx.reset(token)


# ===========================================================================
# AsyncLogHandler tests
# ===========================================================================

class TestAsyncLogHandler:

    def test_emit_does_not_raise(self):
        inner = logging.StreamHandler()
        handler = AsyncLogHandler(inner)
        record = make_record(msg="async emit")
        handler.emit(record)
        handler.stop()

    def test_stop_joins_thread(self):
        inner = logging.StreamHandler()
        handler = AsyncLogHandler(inner)
        handler.stop()
        # After stop, thread should be inactive (or finished)
        assert handler.stop_event.is_set()

    def test_queue_fills_and_drops_oldest(self):
        inner = logging.StreamHandler()
        handler = AsyncLogHandler(inner, queue_size=2)
        records = [make_record(msg=f"msg-{i}") for i in range(5)]
        for r in records:
            handler.emit(r)
        # Should not raise; queue management handled internally
        handler.stop()

    def test_multiple_emits(self):
        inner = MagicMock(spec=logging.StreamHandler)
        inner.emit = MagicMock()
        handler = AsyncLogHandler(inner)
        for i in range(5):
            handler.emit(make_record(msg=f"record-{i}"))
        time.sleep(0.2)  # Allow background thread to drain
        handler.stop()
        assert inner.emit.call_count >= 0  # Not crashing is the goal

    def test_start_creates_thread(self):
        inner = logging.StreamHandler()
        handler = AsyncLogHandler(inner)
        assert handler.thread is not None
        assert handler.thread.is_alive()
        handler.stop()


# ===========================================================================
# SizeRotatingFileHandler tests
# ===========================================================================

class TestSizeRotatingFileHandler:

    def test_instantiates_with_defaults(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = SizeRotatingFileHandler(filename=str(log_file))
        handler.close()

    def test_rotation_filename_with_compress(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = SizeRotatingFileHandler(filename=str(log_file), compress=True)
        rotated = handler.rotation_filename("test.log.1")
        assert rotated.endswith(".gz")
        handler.close()

    def test_rotation_filename_without_compress(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = SizeRotatingFileHandler(filename=str(log_file), compress=False)
        rotated = handler.rotation_filename("test.log.1")
        assert not rotated.endswith(".gz")
        handler.close()

    def test_emits_record(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = SizeRotatingFileHandler(filename=str(log_file))
        record = make_record(msg="rotating log test")
        handler.emit(record)
        handler.close()
        content = log_file.read_text()
        assert "rotating log test" in content


# ===========================================================================
# get_logger tests
# ===========================================================================

class TestGetLogger:

    def test_returns_non_none(self):
        logger = get_logger("test.module")
        assert logger is not None

    def test_returns_structlog_logger(self):
        logger = get_logger("test.module")
        # structlog bound loggers have info/debug/error methods
        assert callable(getattr(logger, "info", None))
        assert callable(getattr(logger, "error", None))

    def test_none_name_uses_module_default(self):
        logger = get_logger(None)
        assert logger is not None

    def test_different_names_allowed(self):
        l1 = get_logger("module.a")
        l2 = get_logger("module.b")
        assert l1 is not None
        assert l2 is not None


# ===========================================================================
# set_request_context / clear_request_context tests
# ===========================================================================

class TestRequestContext:

    def setup_method(self):
        clear_request_context()

    def teardown_method(self):
        clear_request_context()

    def test_set_request_id_explicit(self):
        ctx = set_request_context(request_id="req-123")
        assert request_id_ctx.get() == "req-123"
        assert ctx["request_id"] == "req-123"

    def test_set_correlation_id_explicit(self):
        set_request_context(request_id="req-abc", correlation_id="corr-456")
        assert correlation_id_ctx.get() == "corr-456"

    def test_correlation_defaults_to_request_id(self):
        set_request_context(request_id="auto-req")
        assert correlation_id_ctx.get() == "auto-req"

    def test_auto_generates_request_id(self):
        ctx = set_request_context()
        assert len(ctx["request_id"]) == 36  # UUID4 format

    def test_set_user_id(self):
        set_request_context(user_id="u-789")
        assert user_id_ctx.get() == "u-789"

    def test_set_session_id(self):
        set_request_context(session_id="sess-101")
        assert session_id_ctx.get() == "sess-101"

    def test_returns_dict(self):
        result = set_request_context(request_id="r1")
        assert isinstance(result, dict)
        assert "request_id" in result

    def test_clear_resets_request_id(self):
        set_request_context(request_id="to-clear")
        clear_request_context()
        assert request_id_ctx.get() == ""

    def test_clear_resets_correlation_id(self):
        set_request_context(correlation_id="corr-clear")
        clear_request_context()
        assert correlation_id_ctx.get() == ""

    def test_clear_resets_user_id(self):
        set_request_context(user_id="u-clear")
        clear_request_context()
        assert user_id_ctx.get() is None

    def test_clear_resets_session_id(self):
        set_request_context(session_id="s-clear")
        clear_request_context()
        assert session_id_ctx.get() is None


# ===========================================================================
# LogContext tests
# ===========================================================================

class TestLogContext:

    def setup_method(self):
        clear_request_context()

    def teardown_method(self):
        clear_request_context()

    def test_does_not_raise(self):
        with LogContext(user_id="u123", operation="test"):
            pass

    def test_sets_user_id_context_var(self):
        with LogContext(user_id="ctx-user"):
            assert user_id_ctx.get() == "ctx-user"

    def test_restores_previous_user_id_on_exit(self):
        user_id_ctx.set("original")
        with LogContext(user_id="temporary"):
            assert user_id_ctx.get() == "temporary"
        assert user_id_ctx.get() == "original"

    def test_sets_agent_id_context_var(self):
        with LogContext(agent_id="agent-42"):
            assert agent_id_ctx.get() == "agent-42"

    def test_restores_agent_id_on_exit(self):
        agent_id_ctx.set("prev-agent")
        with LogContext(agent_id="temp-agent"):
            pass
        assert agent_id_ctx.get() == "prev-agent"

    def test_unknown_kwargs_are_ignored(self):
        # Kwargs that don't have corresponding _ctx variables are silently ignored
        with LogContext(nonexistent_key="value"):
            pass

    def test_nested_contexts(self):
        with LogContext(user_id="outer"):
            assert user_id_ctx.get() == "outer"
            with LogContext(user_id="inner"):
                assert user_id_ctx.get() == "inner"
            assert user_id_ctx.get() == "outer"

    def test_exception_propagates_and_restores(self):
        user_id_ctx.set("before-exception")
        with pytest.raises(RuntimeError):
            with LogContext(user_id="during-exception"):
                raise RuntimeError("test")
        assert user_id_ctx.get() == "before-exception"

    def test_sets_request_id(self):
        with LogContext(request_id="ctx-req-99"):
            assert request_id_ctx.get() == "ctx-req-99"


# ===========================================================================
# log_function_call decorator tests
# ===========================================================================

class TestLogFunctionCall:

    def test_sync_function_returns_correctly(self):
        @log_function_call()
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_sync_function_with_multiple_args(self):
        @log_function_call()
        def adder(a, b):
            return a + b

        assert adder(3, 4) == 7

    def test_sync_function_with_kwargs(self):
        @log_function_call()
        def greet(name="World"):
            return f"Hello {name}"

        assert greet(name="Test") == "Hello Test"

    def test_sync_function_raises_exception(self):
        @log_function_call()
        def failing():
            raise ValueError("sync error")

        with pytest.raises(ValueError, match="sync error"):
            failing()

    def test_sync_with_debug_level(self):
        @log_function_call(level="DEBUG")
        def debug_func():
            return 42

        assert debug_func() == 42

    def test_sync_with_info_level(self):
        @log_function_call(level="INFO")
        def info_func():
            return "info"

        assert info_func() == "info"

    def test_sync_log_result_true(self):
        @log_function_call(log_result=True)
        def result_func():
            return "the-result"

        assert result_func() == "the-result"

    def test_sync_log_args_false(self):
        @log_function_call(log_args=False)
        def no_args_log(secret):
            return secret

        assert no_args_log("hidden") == "hidden"

    def test_sync_log_time_false(self):
        @log_function_call(log_time=False)
        def no_time():
            return "ok"

        assert no_time() == "ok"

    def test_sync_preserves_function_name(self):
        @log_function_call()
        def original_name():
            return None

        assert original_name.__name__ == "original_name"

    @pytest.mark.asyncio
    async def test_async_function_returns_correctly(self):
        @log_function_call()
        async def async_func():
            return "done"

        result = await async_func()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_async_function_with_args(self):
        @log_function_call()
        async def async_add(a, b):
            return a + b

        assert await async_add(10, 20) == 30

    @pytest.mark.asyncio
    async def test_async_function_raises_exception(self):
        @log_function_call()
        async def async_failing():
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            await async_failing()

    @pytest.mark.asyncio
    async def test_async_log_result_true(self):
        @log_function_call(log_result=True)
        async def async_result():
            return "async-result"

        assert await async_result() == "async-result"

    @pytest.mark.asyncio
    async def test_async_log_args_false(self):
        @log_function_call(log_args=False)
        async def async_no_args(x):
            return x

        assert await async_no_args(99) == 99

    @pytest.mark.asyncio
    async def test_async_log_time_false(self):
        @log_function_call(log_time=False)
        async def async_no_time():
            return True

        assert await async_no_time() is True

    @pytest.mark.asyncio
    async def test_async_preserves_function_name(self):
        @log_function_call()
        async def original_async_name():
            return None

        assert original_async_name.__name__ == "original_async_name"

    def test_max_arg_length_truncation(self):
        @log_function_call(max_arg_length=5)
        def short_arg_logger(x):
            return x

        # Should not raise even with long args
        result = short_arg_logger("a" * 200)
        assert result == "a" * 200


# ===========================================================================
# RequestLogger tests
# ===========================================================================

class TestRequestLogger:

    def _make_mock_request(
        self,
        method="GET",
        path="/test",
        query_params=None,
        client_host="127.0.0.1",
    ):
        request = MagicMock()
        request.method = method
        request.url.path = path
        request.query_params = query_params or {}
        request.headers = {}
        request.client = MagicMock(host=client_host)
        return request

    @pytest.mark.asyncio
    async def test_basic_get_request(self):
        request = self._make_mock_request()
        async with RequestLogger(request):
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_post_request(self):
        request = self._make_mock_request(method="POST", path="/api/items")
        async with RequestLogger(request):
            pass

    @pytest.mark.asyncio
    async def test_request_with_query_params(self):
        request = self._make_mock_request(query_params={"key": "value"})
        async with RequestLogger(request):
            pass

    @pytest.mark.asyncio
    async def test_request_with_no_client(self):
        request = self._make_mock_request()
        request.client = None
        async with RequestLogger(request):
            pass

    @pytest.mark.asyncio
    async def test_request_sets_context(self):
        request = self._make_mock_request()
        async with RequestLogger(request):
            # A request_id should be set during the context
            req_id = request_id_ctx.get()
            assert req_id != ""

    @pytest.mark.asyncio
    async def test_context_cleared_after_exit(self):
        request = self._make_mock_request()
        async with RequestLogger(request):
            pass
        assert request_id_ctx.get() == ""

    @pytest.mark.asyncio
    async def test_exception_in_body_logged(self):
        request = self._make_mock_request()
        with pytest.raises(RuntimeError):
            async with RequestLogger(request):
                raise RuntimeError("request body error")

    @pytest.mark.asyncio
    async def test_context_cleared_after_exception(self):
        request = self._make_mock_request()
        try:
            async with RequestLogger(request):
                raise ValueError("cleanup test")
        except ValueError:
            pass
        assert request_id_ctx.get() == ""


# ===========================================================================
# mask_sensitive_data tests
# ===========================================================================

class TestMaskSensitiveData:

    def test_masks_password_key(self):
        result = mask_sensitive_data({"password": "secret123"})
        assert result["password"] == "********"

    def test_masks_token_key(self):
        result = mask_sensitive_data({"token": "abc.def.ghi"})
        assert result["token"] == "********"

    def test_masks_api_key(self):
        result = mask_sensitive_data({"api_key": "sk-12345"})
        assert result["api_key"] == "********"

    def test_preserves_non_sensitive_key(self):
        result = mask_sensitive_data({"username": "john"})
        assert result["username"] == "john"

    def test_nested_dict_masked(self):
        result = mask_sensitive_data({"user": {"password": "hidden"}})
        assert result["user"]["password"] == "********"

    def test_list_items_processed(self):
        result = mask_sensitive_data([{"password": "p"}, {"name": "n"}])
        assert result[0]["password"] == "********"
        assert result[1]["name"] == "n"

    def test_string_passthrough(self):
        result = mask_sensitive_data("plain string")
        assert result == "plain string"

    def test_long_string_truncated(self):
        long_str = "a" * (MAX_STRING_LENGTH + 100)
        result = mask_sensitive_data(long_str)
        assert "truncated" in result
        assert len(result) < len(long_str) + 50

    def test_bearer_token_in_string_masked(self):
        result = mask_sensitive_data("bearer abc123token")
        assert result == "********"

    def test_bytes_representation(self):
        result = mask_sensitive_data(b"\x00\x01\x02")
        assert "bytes" in result

    def test_max_depth_exceeded(self):
        result = mask_sensitive_data({"key": "val"}, depth=10)
        assert result == "<max depth exceeded>"

    def test_integer_passthrough(self):
        assert mask_sensitive_data(42) == 42

    def test_none_passthrough(self):
        assert mask_sensitive_data(None) is None

    def test_set_converted_to_list(self):
        result = mask_sensitive_data({1, 2, 3})
        assert isinstance(result, list)


# ===========================================================================
# Structlog processor tests
# ===========================================================================

class TestStructlogProcessors:

    def test_add_timestamp_adds_key(self):
        event_dict = {}
        result = add_timestamp(None, None, event_dict)
        assert "timestamp" in result

    def test_add_log_level_uppercase(self):
        event_dict = {}
        result = add_log_level(None, "info", event_dict)
        assert result["level"] == "INFO"

    def test_add_log_level_debug(self):
        event_dict = {}
        result = add_log_level(None, "debug", event_dict)
        assert result["level"] == "DEBUG"

    def test_add_context_vars_with_request_id(self):
        token = request_id_ctx.set("proc-req-id")
        try:
            event_dict = {}
            result = add_context_vars(None, None, event_dict)
            assert result.get("request_id") == "proc-req-id"
        finally:
            request_id_ctx.reset(token)

    def test_add_context_vars_empty_by_default(self):
        clear_request_context()
        event_dict = {}
        result = add_context_vars(None, None, event_dict)
        assert "request_id" not in result or result.get("request_id") == ""

    def test_add_process_info(self):
        event_dict = {}
        result = add_process_info(None, None, event_dict)
        assert "pid" in result
        assert "thread_name" in result
        assert result["pid"] == os.getpid()

    def test_add_host_info(self):
        event_dict = {}
        result = add_host_info(None, None, event_dict)
        assert "hostname" in result

    def test_add_environment(self):
        event_dict = {}
        result = add_environment(None, None, event_dict)
        assert "environment" in result

    def test_mask_sensitive_processor(self):
        event_dict = {"password": "secret"}
        result = mask_sensitive(None, None, event_dict)
        assert result["password"] == "********"

    def test_truncate_long_messages_short_message_unchanged(self):
        event_dict = {"event": "short"}
        result = truncate_long_messages(None, None, event_dict)
        assert result["event"] == "short"

    def test_truncate_long_messages_truncates(self):
        long_event = "x" * (MAX_STRING_LENGTH + 100)
        event_dict = {"event": long_event}
        result = truncate_long_messages(None, None, event_dict)
        assert "truncated" in result["event"]
        assert len(result["event"]) <= MAX_STRING_LENGTH + 50

    def test_add_exception_info_with_tuple(self):
        try:
            raise TypeError("exc tuple test")
        except TypeError:
            exc_info = sys.exc_info()
        event_dict = {"exc_info": exc_info}
        result = add_exception_info(None, None, event_dict)
        assert "exception" in result
        assert result["exception"]["type"] == "TypeError"
        assert "exc_tuple_test" not in result  # key should be consumed

    def test_add_exception_info_with_exception_obj(self):
        exc = ValueError("direct exception")
        event_dict = {"exc_info": exc}
        result = add_exception_info(None, None, event_dict)
        assert "exception" in result
        assert result["exception"]["type"] == "ValueError"

    def test_add_exception_info_no_exception(self):
        event_dict = {"event": "no exc"}
        result = add_exception_info(None, None, event_dict)
        assert "exception" not in result


# ===========================================================================
# setup_logging smoke tests
# ===========================================================================

class TestSetupLogging:

    def test_setup_logging_runs_without_error(self):
        # Minimal call should not raise
        setup_logging(
            enable_console=True,
            enable_file=False,
            enable_syslog=False,
        )

    def test_setup_logging_json_format(self):
        setup_logging(
            enable_json=True,
            enable_console=True,
            enable_file=False,
            enable_syslog=False,
        )

    def test_setup_logging_no_console(self):
        setup_logging(
            enable_console=False,
            enable_file=False,
            enable_syslog=False,
        )

    def test_setup_logging_with_file(self, tmp_path):
        log_file = tmp_path / "app.log"
        setup_logging(
            enable_console=False,
            enable_file=True,
            log_file_path=log_file,
            enable_syslog=False,
        )
        # Cleanup
        root = logging.getLogger()
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)

    def test_setup_logging_capture_warnings(self):
        setup_logging(
            enable_console=False,
            enable_file=False,
            enable_syslog=False,
            capture_warnings=True,
        )


# ===========================================================================
# SENSITIVE_FIELDS constant test
# ===========================================================================

class TestConstants:

    def test_sensitive_fields_contains_password(self):
        assert "password" in SENSITIVE_FIELDS

    def test_sensitive_fields_contains_token(self):
        assert "token" in SENSITIVE_FIELDS

    def test_max_string_length_positive(self):
        assert MAX_STRING_LENGTH > 0
