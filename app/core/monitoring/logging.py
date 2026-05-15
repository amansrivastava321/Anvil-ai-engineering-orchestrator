"""
Enterprise-grade structured logging system with:
- Multiple handlers (console, file, syslog, remote)
- Structured JSON logging for log aggregation
- Automatic context injection (request ID, user, session)
- Log rotation with compression
- Asynchronous logging for performance
- Log level management per module
- Sensitive data masking
- Performance metrics in logs
- Color-coded console output for development

This module provides the foundation for all application observability.
Security Level: HIGH - Handles sanitization of sensitive data in logs
"""

import sys
import os
import json
import logging
import logging.handlers
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Callable
from datetime import datetime, timezone
from functools import wraps
import traceback
from contextvars import ContextVar
import uuid

import structlog
from structlog.types import Processor, EventDict, WrappedLogger

from app.core.config.settings import settings, LogLevel, LogFormat, Environment


# ============================================================================
# Context Variables for Request Tracking
# ============================================================================

# These context variables maintain state across async operations
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
agent_id_ctx: ContextVar[Optional[str]] = ContextVar("agent_id", default=None)
trace_id_ctx: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


# ============================================================================
# Sensitive Data Patterns for Masking
# ============================================================================

# Patterns to mask in logs
SENSITIVE_FIELDS = {
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "auth", "credentials", "private_key", "privatekey",
    "access_token", "accesstoken", "refresh_token", "refreshtoken",
    "ssn", "credit_card", "creditcard", "card_number", "cardnumber",
    "cvv", "cvc", "pin", "security_code",
}

SENSITIVE_HEADERS = {
    "authorization", "x-api-key", "x-auth-token", "cookie",
    "set-cookie", "x-csrf-token", "x-forwarded-for",
}

# Maximum length for string values in logs
MAX_STRING_LENGTH = 10000

# Maximum depth for nested objects
MAX_DICT_DEPTH = 5


# ============================================================================
# Data Sanitization
# ============================================================================

def mask_sensitive_data(data: Any, depth: int = 0) -> Any:
    """
    Recursively mask sensitive data in logs.
    
    Args:
        data: Data to sanitize
        depth: Current recursion depth
        
    Returns:
        Sanitized data
    """
    if depth > MAX_DICT_DEPTH:
        return "<max depth exceeded>"
    
    # Handle dictionaries
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key is sensitive
            if any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS):
                sanitized[key] = "********"
            elif key_lower in SENSITIVE_HEADERS:
                sanitized[key] = "********"
            else:
                sanitized[key] = mask_sensitive_data(value, depth + 1)
        return sanitized
    
    # Handle lists
    elif isinstance(data, (list, tuple, set)):
        return [mask_sensitive_data(item, depth + 1) for item in data]
    
    # Handle strings
    elif isinstance(data, str):
        # Truncate long strings
        if len(data) > MAX_STRING_LENGTH:
            return data[:MAX_STRING_LENGTH] + f"... [truncated {len(data) - MAX_STRING_LENGTH} chars]"
        
        # Mask potential secrets in strings (heuristic)
        if any(pattern in data.lower() for pattern in ["bearer ", "token=", "api_key="]):
            return "********"
        
        return data
    
    # Handle bytes
    elif isinstance(data, bytes):
        return f"<bytes: {len(data)} bytes>"
    
    # Return other types as-is
    return data


# ============================================================================
# Custom Processors
# ============================================================================

def add_timestamp(_, __, event_dict: EventDict) -> EventDict:
    """Add ISO 8601 timestamp to log event."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_log_level(_, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event."""
    event_dict["level"] = method_name.upper()
    return event_dict


def add_context_vars(_, __, event_dict: EventDict) -> EventDict:
    """Add context variables to log event."""
    request_id = request_id_ctx.get()
    if request_id:
        event_dict["request_id"] = request_id
    
    correlation_id = correlation_id_ctx.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    
    user_id = user_id_ctx.get()
    if user_id:
        event_dict["user_id"] = user_id
    
    session_id = session_id_ctx.get()
    if session_id:
        event_dict["session_id"] = session_id
    
    agent_id = agent_id_ctx.get()
    if agent_id:
        event_dict["agent_id"] = agent_id
    
    trace_id = trace_id_ctx.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    
    return event_dict


def add_process_info(_, __, event_dict: EventDict) -> EventDict:
    """Add process and thread information."""
    event_dict["pid"] = os.getpid()
    event_dict["thread_name"] = threading.current_thread().name
    return event_dict


def add_host_info(_, __, event_dict: EventDict) -> EventDict:
    """Add host information."""
    event_dict["hostname"] = os.uname().nodename if hasattr(os, "uname") else "unknown"
    return event_dict


def add_environment(_, __, event_dict: EventDict) -> EventDict:
    """Add application environment."""
    env = settings.environment
    event_dict["environment"] = env.value if hasattr(env, "value") else str(env)
    return event_dict


def mask_sensitive(_, __, event_dict: EventDict) -> EventDict:
    """Mask sensitive data in event dict."""
    return mask_sensitive_data(event_dict)


def truncate_long_messages(_, __, event_dict: EventDict) -> EventDict:
    """Truncate long messages."""
    event = event_dict.get("event", "")
    if isinstance(event, str) and len(event) > MAX_STRING_LENGTH:
        event_dict["event"] = event[:MAX_STRING_LENGTH] + "... [truncated]"
    return event_dict


def add_exception_info(_, __, event_dict: EventDict) -> EventDict:
    """Format exception information."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        if isinstance(exc_info, tuple):
            event_dict["exception"] = {
                "type": exc_info[0].__name__ if exc_info[0] else None,
                "message": str(exc_info[1]) if exc_info[1] else None,
                "traceback": (
                    "".join(traceback.format_tb(exc_info[2]))
                    if exc_info[2]
                    else None
                ),
            }
        elif isinstance(exc_info, BaseException):
            event_dict["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
                "traceback": traceback.format_exc(),
            }
    return event_dict


# ============================================================================
# Formatters
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    Outputs logs as JSON objects suitable for log aggregation systems.
    """
    
    def __init__(self):
        super().__init__()
        self.default_fields = {
            "service": settings.app_name,
            "version": settings.app_version,
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            **self.default_fields,
        }
        
        # Add structured fields if available
        if hasattr(record, "event_dict"):
            log_entry.update(record.event_dict)
        
        # Add extra attributes
        if hasattr(record, "__dict__"):
            for key in record.__dict__:
                if key not in {
                    "args", "asctime", "created", "exc_info", "exc_text",
                    "filename", "funcName", "levelname", "levelno",
                    "lineno", "module", "msecs", "message", "msg",
                    "name", "pathname", "process", "processName",
                    "relativeCreated", "stack_info", "thread", "threadName",
                }:
                    log_entry[key] = getattr(record, key)
        
        # Handle exceptions
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        # Mask sensitive data
        log_entry = mask_sensitive_data(log_entry)
        
        try:
            return json.dumps(log_entry, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback if JSON serialization fails
            return json.dumps({
                "timestamp": log_entry.get("timestamp"),
                "level": log_entry.get("level"),
                "message": "Log serialization failed",
                "original_message": str(record.getMessage())[:200],
            })


class ConsoleFormatter(logging.Formatter):
    """
    Colorized console formatter for development.
    Uses ANSI color codes for better readability.
    """
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
        "RESET": "\033[0m",       # Reset
        "BOLD": "\033[1m",        # Bold
        "DIM": "\033[2m",         # Dim
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for log level
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Format log level with fixed width
        level = f"{level_color}{record.levelname:8}{reset}"
        
        # Format logger name with dim color
        logger_name = f"{self.COLORS['DIM']}{record.name}{reset}"
        
        # Format message
        message = record.getMessage()
        
        # Add context information if available
        context_parts = []
        request_id = request_id_ctx.get()
        if request_id:
            context_parts.append(f"req={request_id[:8]}")
        
        agent_id = agent_id_ctx.get()
        if agent_id:
            context_parts.append(f"agent={agent_id}")
        
        context_str = ""
        if context_parts:
            context_str = f" {self.COLORS['DIM']}[{' | '.join(context_parts)}]{reset}"
        
        # Build format string
        log_line = f"{self.COLORS['DIM']}{timestamp}{reset} {level} [{logger_name}]{context_str} {message}"
        
        # Add exception if present
        if record.exc_info:
            log_line += f"\n{self.COLORS['DIM']}{self.formatException(record.exc_info)}{reset}"
        
        # Add stack info if present
        if record.stack_info:
            log_line += f"\n{self.COLORS['DIM']}{self.formatStack(record.stack_info)}{reset}"
        
        return log_line


class KeyValueFormatter(logging.Formatter):
    """
    Key=value formatter for easy parsing by log aggregators.
    Format: timestamp level logger key1=value1 key2=value2 message
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as key=value pairs."""
        parts = [
            f"timestamp={datetime.fromtimestamp(record.created).isoformat()}",
            f"level={record.levelname}",
            f"logger={record.name}",
        ]
        
        # Add context variables
        request_id = request_id_ctx.get()
        if request_id:
            parts.append(f"request_id={request_id}")
        
        user_id = user_id_ctx.get()
        if user_id:
            parts.append(f"user_id={user_id}")
        
        # Add message
        parts.append(f"message=\"{record.getMessage()}\"")
        
        # Add exception if present
        if record.exc_info:
            parts.append(f"exception=\"{str(record.exc_info[1])}\"")
        
        return " ".join(parts)


# ============================================================================
# Handlers
# ============================================================================

class AsyncLogHandler(logging.Handler):
    """
    Asynchronous log handler that emits logs in a background thread.
    Prevents logging from blocking the main application thread.
    """
    
    def __init__(self, handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.handler = handler
        self.queue: List[logging.LogRecord] = []
        self.queue_size = queue_size
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.start()
    
    def start(self):
        """Start the background logging thread."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._process_queue, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the background logging thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def _process_queue(self):
        """Process log records from queue."""
        while not self.stop_event.is_set():
            record = None
            try:
                with self.lock:
                    if self.queue:
                        record = self.queue.pop(0)
            except Exception:
                pass

            if record is not None:
                try:
                    self.handler.emit(record)
                except Exception:
                    pass
            else:
                # Sleep WITHOUT holding the lock to avoid starving emitters
                time.sleep(0.01)
    
    def emit(self, record: logging.LogRecord):
        """Add record to queue."""
        try:
            with self.lock:
                if len(self.queue) < self.queue_size:
                    self.queue.append(record)
                else:
                    # Queue full - drop oldest record
                    self.queue.pop(0)
                    self.queue.append(record)
        except Exception:
            self.handleError(record)


class SizeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced rotating file handler with compression support.
    """
    
    def __init__(
        self,
        filename: Union[str, Path],
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        compress: bool = True,
        encoding: str = "utf-8",
    ):
        super().__init__(
            filename=str(filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )
        self.compress = compress
    
    def rotation_filename(self, default_name: str) -> str:
        """Override rotation filename to add compression extension."""
        if self.compress:
            return default_name + ".gz"
        return default_name
    
    def rotate(self, source: str, dest: str) -> None:
        """Rotate and compress the log file."""
        super().rotate(source, dest)
        
        if self.compress and dest.endswith(".gz"):
            import gzip
            import shutil
            
            # Compress the rotated file
            try:
                with open(dest.rstrip(".gz"), "rb") as f_in:
                    with gzip.open(dest, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(dest.rstrip(".gz"))
            except Exception as e:
                # Log compression error but don't crash
                print(f"Failed to compress log file: {e}", file=sys.stderr)


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(
    log_level: Optional[LogLevel] = None,
    log_format: Optional[LogFormat] = None,
    log_file_path: Optional[Path] = None,
    enable_json: Optional[bool] = None,
    enable_console: Optional[bool] = None,
    enable_file: Optional[bool] = None,
    enable_syslog: Optional[bool] = None,
    async_handlers: bool = False,
    capture_warnings: bool = True,
) -> None:
    """
    Configure application-wide logging with multiple handlers.
    
    Args:
        log_level: Log level (default from settings)
        log_format: Log format (default from settings)
        log_file_path: Path to log file
        enable_json: Enable JSON logging
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_syslog: Enable syslog logging
        async_handlers: Use async handlers for better performance
        capture_warnings: Capture Python warnings as log messages
    """
    # Use settings if not specified
    log_level = log_level or settings.monitoring.log_level
    log_format = log_format or settings.monitoring.log_format
    log_file_path = log_file_path or settings.monitoring.log_file_path
    enable_json = enable_json if enable_json is not None else (log_format == LogFormat.JSON)
    enable_console = enable_console if enable_console is not None else True
    enable_file = enable_file if enable_file is not None else bool(log_file_path)
    enable_syslog = enable_syslog if enable_syslog is not None else settings.is_production
    
    # Convert log level
    level = getattr(logging, log_level.value if isinstance(log_level, LogLevel) else str(log_level).upper())
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # ========================================
    # Console Handler
    # ========================================
    if enable_console:
        if enable_json:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            if sys.stdout.isatty():
                console_handler.setFormatter(ConsoleFormatter())
            else:
                console_handler.setFormatter(KeyValueFormatter())
        
        console_handler.setLevel(level)
        
        if async_handlers and not settings.is_testing:
            console_handler = AsyncLogHandler(console_handler)
        
        root_logger.addHandler(console_handler)
    
    # ========================================
    # File Handler
    # ========================================
    if enable_file and log_file_path:
        # Ensure log directory exists
        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Main log file with rotation
        file_handler = SizeRotatingFileHandler(
            filename=log_file_path,
            max_bytes=settings.monitoring.log_max_size,
            backup_count=settings.monitoring.log_backup_count,
            compress=True,
        )
        file_handler.setFormatter(JSONFormatter() if enable_json else KeyValueFormatter())
        file_handler.setLevel(level)
        
        if async_handlers and not settings.is_testing:
            file_handler = AsyncLogHandler(file_handler)
        
        root_logger.addHandler(file_handler)
        
        # Error log file (errors only)
        error_log_path = log_file_path.parent / f"error{log_file_path.suffix}"
        error_handler = SizeRotatingFileHandler(
            filename=error_log_path,
            max_bytes=settings.monitoring.log_max_size,
            backup_count=settings.monitoring.log_backup_count,
            compress=True,
        )
        error_handler.setFormatter(JSONFormatter() if enable_json else KeyValueFormatter())
        error_handler.setLevel(logging.ERROR)
        
        if async_handlers and not settings.is_testing:
            error_handler = AsyncLogHandler(error_handler)
        
        root_logger.addHandler(error_handler)
        
        # Access log file (for HTTP access logs)
        access_log_path = log_file_path.parent / f"access{log_file_path.suffix}"
        access_handler = SizeRotatingFileHandler(
            filename=access_log_path,
            max_bytes=settings.monitoring.log_max_size,
            backup_count=settings.monitoring.log_backup_count,
            compress=True,
        )
        access_handler.setFormatter(JSONFormatter() if enable_json else KeyValueFormatter())
        access_handler.setLevel(logging.INFO)
        
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.addHandler(access_handler)
        access_logger.propagate = False
    
    # ========================================
    # Syslog Handler (Production)
    # ========================================
    if enable_syslog:
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address="/dev/log" if sys.platform != "win32" else ("localhost", 514)
            )
            syslog_handler.setFormatter(KeyValueFormatter())
            syslog_handler.setLevel(logging.WARNING)
            root_logger.addHandler(syslog_handler)
        except Exception as e:
            print(f"Failed to setup syslog handler: {e}", file=sys.stderr)
    
    # ========================================
    # Configure Structlog
    # ========================================
    
    # Build processor chain
    processors: List[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_log_level,
        add_context_vars,
        add_process_info,
        add_host_info,
        add_environment,
        mask_sensitive,
        truncate_long_messages,
        add_exception_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add appropriate renderer
    if enable_json:
        processors.append(structlog.processors.JSONRenderer(serializer=json.dumps))
    elif sys.stdout.isatty():
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # ========================================
    # Configure Third-Party Loggers
    # ========================================
    
    # Silence noisy third-party loggers
    for logger_name in [
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "aiohttp",
        "aiosqlite",
        "sqlalchemy.engine",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Set specific loggers to INFO
    for logger_name in [
        "uvicorn",
        "fastapi",
        "app",
    ]:
        logging.getLogger(logger_name).setLevel(level)
    
    # ========================================
    # Capture Warnings
    # ========================================
    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)
    
    # ========================================
    # Log Startup
    # ========================================
    logger = structlog.get_logger(__name__)
    logger.info(
        "Logging configured",
        level=log_level.value if isinstance(log_level, LogLevel) else log_level,
        format=log_format.value if isinstance(log_format, LogFormat) else log_format,
        console=enable_console,
        file=enable_file,
        json=enable_json,
        async_handlers=async_handlers,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("message", key="value")
    """
    return structlog.get_logger(name or __name__)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Set context variables for the current request/async context.
    Returns the set values for use in context managers.
    
    Usage:
        with set_request_context() as ctx:
            logger.info("Processing request")
    """
    context = {}
    
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_ctx.set(request_id)
    context["request_id"] = request_id
    
    if correlation_id is None:
        correlation_id = request_id
    correlation_id_ctx.set(correlation_id)
    context["correlation_id"] = correlation_id
    
    if user_id:
        user_id_ctx.set(user_id)
        context["user_id"] = user_id
    
    if session_id:
        session_id_ctx.set(session_id)
        context["session_id"] = session_id
    
    return context


def clear_request_context() -> None:
    """Clear context variables after request completion."""
    request_id_ctx.set("")
    correlation_id_ctx.set("")
    user_id_ctx.set(None)
    session_id_ctx.set(None)
    agent_id_ctx.set(None)
    trace_id_ctx.set(None)


class LogContext:
    """
    Context manager for temporary log context.
    
    Usage:
        with LogContext(user_id="123", agent_id="agent-1"):
            logger.info("Processing with context")
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.previous = {}
    
    def __enter__(self):
        # Save current values
        for key, value in self.kwargs.items():
            ctx_var = globals().get(f"{key}_ctx")
            if ctx_var:
                self.previous[key] = ctx_var.get()
                ctx_var.set(value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous values
        for key, value in self.previous.items():
            ctx_var = globals().get(f"{key}_ctx")
            if ctx_var:
                ctx_var.set(value)


def log_function_call(
    level: str = "DEBUG",
    log_args: bool = True,
    log_result: bool = False,
    log_time: bool = True,
    max_arg_length: int = 100,
):
    """
    Decorator to log function calls with arguments and timing.
    
    Usage:
        @log_function_call(level="INFO", log_time=True)
        async def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Build argument string
            arg_str = ""
            if log_args:
                func_args = []
                for i, arg in enumerate(args):
                    if i == 0 and hasattr(arg, "__class__"):  # Skip 'self'
                        continue
                    arg_str_val = str(arg)
                    if len(arg_str_val) > max_arg_length:
                        arg_str_val = arg_str_val[:max_arg_length] + "..."
                    func_args.append(arg_str_val)
                for key, value in kwargs.items():
                    val_str = str(value)
                    if len(val_str) > max_arg_length:
                        val_str = val_str[:max_arg_length] + "..."
                    func_args.append(f"{key}={val_str}")
                arg_str = ", ".join(func_args)
            
            getattr(logger, level.lower(), logger.debug)(
                f"Calling {func.__name__}",
                function=func.__name__,
                args=arg_str if arg_str else None,
            )
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                
                if log_time:
                    elapsed = time.time() - start_time
                    getattr(logger, level.lower(), logger.debug)(
                        f"Completed {func.__name__}",
                        function=func.__name__,
                        elapsed_ms=round(elapsed * 1000, 2),
                    )
                
                if log_result and result is not None:
                    result_str = str(result)
                    if len(result_str) > max_arg_length:
                        result_str = result_str[:max_arg_length] + "..."
                    getattr(logger, level.lower(), logger.debug)(
                        f"Result from {func.__name__}",
                        result=result_str,
                    )
                
                return result
                
            except Exception as e:
                if log_time:
                    elapsed = time.time() - start_time
                    logger.error(
                        f"Failed {func.__name__}",
                        function=func.__name__,
                        elapsed_ms=round(elapsed * 1000, 2),
                        error=str(e),
                        exc_info=True,
                    )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            arg_str = ""
            if log_args:
                func_args = []
                for i, arg in enumerate(args):
                    if i == 0 and hasattr(arg, "__class__"):
                        continue
                    arg_str_val = str(arg)
                    if len(arg_str_val) > max_arg_length:
                        arg_str_val = arg_str_val[:max_arg_length] + "..."
                    func_args.append(arg_str_val)
                for key, value in kwargs.items():
                    val_str = str(value)
                    if len(val_str) > max_arg_length:
                        val_str = val_str[:max_arg_length] + "..."
                    func_args.append(f"{key}={val_str}")
                arg_str = ", ".join(func_args)
            
            getattr(logger, level.lower(), logger.debug)(
                f"Calling {func.__name__}",
                function=func.__name__,
                args=arg_str if arg_str else None,
            )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                if log_time:
                    elapsed = time.time() - start_time
                    getattr(logger, level.lower(), logger.debug)(
                        f"Completed {func.__name__}",
                        function=func.__name__,
                        elapsed_ms=round(elapsed * 1000, 2),
                    )
                
                return result
                
            except Exception as e:
                if log_time:
                    elapsed = time.time() - start_time
                    logger.error(
                        f"Failed {func.__name__}",
                        function=func.__name__,
                        elapsed_ms=round(elapsed * 1000, 2),
                        error=str(e),
                        exc_info=True,
                    )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class RequestLogger:
    """
    Middleware-compatible request logger for FastAPI.
    
    Usage:
        @app.middleware("http")
        async def log_requests(request, call_next):
            async with RequestLogger(request) as logger:
                response = await call_next(request)
                return response
    """
    
    def __init__(self, request):
        self.request = request
        self.start_time = None
        self.logger = get_logger("api.requests")
    
    async def __aenter__(self):
        self.start_time = time.time()
        
        # Set request context
        set_request_context(
            request_id=str(uuid.uuid4()),
        )
        
        # Log request
        self.logger.info(
            "Request started",
            method=self.request.method,
            path=self.request.url.path,
            query=str(self.request.query_params) if self.request.query_params else None,
            client=self.request.client.host if self.request.client else None,
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if exc_type:
            self.logger.error(
                "Request failed",
                method=self.request.method,
                path=self.request.url.path,
                elapsed_ms=round(elapsed * 1000, 2),
                error=str(exc_val),
                exc_info=(exc_type, exc_val, exc_tb),
            )
        else:
            self.logger.info(
                "Request completed",
                method=self.request.method,
                path=self.request.url.path,
                elapsed_ms=round(elapsed * 1000, 2),
            )
        
        # Clear context
        clear_request_context()


# Initialize logging when module is imported
if not settings.is_testing:
    setup_logging()

logger = get_logger(__name__)
logger.info("Logging module initialized successfully")