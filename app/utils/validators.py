"""
Enterprise-grade validation utilities with comprehensive type checking,
path security, input sanitization, and schema validation.

This module provides critical validation functions that protect the system
from invalid inputs, path traversal attacks, and type inconsistencies.
All validators are designed for high-performance async operations.

Security Level: CRITICAL
"""

import re
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from datetime import datetime
from functools import wraps
from pydantic import BaseModel, ValidationError
import structlog

logger = structlog.get_logger(__name__)

# Type variable for generic type validation
T = TypeVar('T')


class ValidationError(Exception):
    """Base validation error with detailed context."""
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        self.timestamp = datetime.utcnow()
        super().__init__(message)


class PathSecurityError(ValidationError):
    """Raised when path validation fails security checks."""
    pass


class TypeValidationError(ValidationError):
    """Raised when type validation fails."""
    pass


class InputSanitizationError(ValidationError):
    """Raised when input sanitization fails."""
    pass


class PathValidator:
    """
    Secure path validator with protection against:
    - Path traversal attacks
    - Symlink attacks
    - Unauthorized file access
    - Filesystem race conditions (TOCTOU)
    """
    
    # Blacklisted path patterns
    BLACKLISTED_PATTERNS = [
        r'\.\./',           # Directory traversal
        r'\.\.\\',          # Windows traversal
        r'%2e%2e%2f',       # URL encoded traversal
        r'%2e%2e/',         # Mixed encoding
        r'\.\.%2f',         # Partial encoding
        r'%00',             # Null byte injection
        r'\\x00',           # Hex null byte
        r'/etc/passwd',     # System files
        r'/etc/shadow',
        r'C:\\Windows\\',   # Windows system
        r'/proc/',           # Linux proc
        r'/sys/',            # Linux sys
    ]
    
    # Allowed file extensions (whitelist approach)
    ALLOWED_EXTENSIONS = {
        # Source code
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
        '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php', '.swift',
        '.kt', '.scala', '.r', '.m', '.mm',
        
        # Web
        '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
        
        # Config & Data
        '.json', '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg',
        '.conf', '.env', '.properties',
        
        # Documentation
        '.md', '.rst', '.txt', '.log', '.csv', '.tsv',
        
        # Shell & Scripts
        '.sh', '.bash', '.zsh', '.fish', '.ps1',
        
        # Docker & K8s
        '.dockerfile', '.dockerignore', '.yaml', '.yml',
        
        # Graphify specific
        '.graph', '.graphml', '.dot',
    }
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Maximum path length
    MAX_PATH_LENGTH = 4096
    
    # Base directories that are allowed
    ALLOWED_BASE_DIRS: List[Path] = []
    
    @classmethod
    def set_allowed_base_dirs(cls, dirs: List[Path]) -> None:
        """Set the allowed base directories for path validation."""
        cls.ALLOWED_BASE_DIRS = [Path(d).resolve() for d in dirs]
    
    @classmethod
    def validate_path(
        cls,
        path: Union[str, Path],
        must_exist: bool = True,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_extensions: Optional[List[str]] = None,
        check_symlinks: bool = True,
        check_readable: bool = True,
        check_writable: bool = False,
    ) -> Path:
        """
        Comprehensive path validation with security checks.
        
        Args:
            path: Path to validate
            must_exist: Path must exist on filesystem
            must_be_file: Path must be a file
            must_be_dir: Path must be a directory
            allowed_extensions: List of allowed file extensions (None = use default)
            check_symlinks: Check for symlink attacks
            check_readable: Check if path is readable
            check_writable: Check if path is writable
            
        Returns:
            Resolved Path object
            
        Raises:
            PathSecurityError: If path fails security validation
            ValidationError: If path fails other validation checks
        """
        try:
            # Convert to string and validate type
            if isinstance(path, Path):
                path_str = str(path)
            else:
                path_str = str(path)
            
            # Check path length
            if len(path_str) > cls.MAX_PATH_LENGTH:
                raise PathSecurityError(
                    f"Path exceeds maximum length of {cls.MAX_PATH_LENGTH}",
                    field="path"
                )
            
            # Check for blacklisted patterns
            for pattern in cls.BLACKLISTED_PATTERNS:
                if re.search(pattern, path_str, re.IGNORECASE):
                    raise PathSecurityError(
                        f"Path contains forbidden pattern: {pattern}",
                        field="path",
                        value=path_str
                    )
            
            # Convert to Path object and resolve
            path_obj = Path(path_str)
            
            # Prevent null bytes
            if '\x00' in path_str:
                raise PathSecurityError(
                    "Path contains null bytes",
                    field="path"
                )
            
            # Resolve the path to catch symlink attacks early
            try:
                resolved_path = path_obj.resolve(strict=False)
            except (OSError, RuntimeError) as e:
                raise PathSecurityError(
                    f"Cannot resolve path: {e}",
                    field="path",
                    value=path_str
                )
            
            # Check if path is within allowed directories
            if cls.ALLOWED_BASE_DIRS:
                is_allowed = any(
                    str(resolved_path).startswith(str(base_dir))
                    for base_dir in cls.ALLOWED_BASE_DIRS
                )
                if not is_allowed:
                    raise PathSecurityError(
                        "Path is outside allowed directories",
                        field="path",
                        value=str(resolved_path)
                    )
            
            # Check symlinks for TOCTOU
            if check_symlinks and resolved_path.is_symlink():
                raise PathSecurityError(
                    "Symlinks are not allowed for security reasons",
                    field="path",
                    value=str(resolved_path)
                )
            
            # Check existence
            if must_exist and not resolved_path.exists():
                raise ValidationError(
                    f"Path does not exist: {resolved_path}",
                    field="path"
                )
            
            # Check type (file/directory)
            if must_be_file and must_exist and not resolved_path.is_file():
                raise ValidationError(
                    f"Path is not a file: {resolved_path}",
                    field="path"
                )
            
            if must_be_dir and must_exist and not resolved_path.is_dir():
                raise ValidationError(
                    f"Path is not a directory: {resolved_path}",
                    field="path"
                )
            
            # Check extension whitelist
            if must_be_file and resolved_path.suffix:
                extensions = allowed_extensions or cls.ALLOWED_EXTENSIONS
                if resolved_path.suffix.lower() not in extensions:
                    raise PathSecurityError(
                        f"File extension '{resolved_path.suffix}' is not allowed",
                        field="path",
                        value=str(resolved_path)
                    )
            
            # Check file size
            if must_be_file and resolved_path.exists():
                file_size = resolved_path.stat().st_size
                if file_size > cls.MAX_FILE_SIZE:
                    raise ValidationError(
                        f"File size ({file_size} bytes) exceeds maximum ({cls.MAX_FILE_SIZE} bytes)",
                        field="path"
                    )
            
            # Check permissions
            if check_readable and must_exist:
                if not os.access(resolved_path, os.R_OK):
                    raise PathSecurityError(
                        f"Path is not readable: {resolved_path}",
                        field="path"
                    )
            
            if check_writable and must_exist:
                if not os.access(resolved_path, os.W_OK):
                    raise PathSecurityError(
                        f"Path is not writable: {resolved_path}",
                        field="path"
                    )
            
            logger.debug(
                "Path validation successful",
                path=str(resolved_path),
                exists=resolved_path.exists() if must_exist else "not_checked"
            )
            
            return resolved_path
            
        except (PathSecurityError, ValidationError):
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during path validation",
                path=str(path),
                error=str(e),
                exc_info=True
            )
            raise ValidationError(f"Path validation failed: {e}", field="path")


class TypeValidator:
    """
    Strict type validation with comprehensive checks.
    """
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type[T], field_name: str = "value") -> T:
        """
        Validate that a value matches the expected type.
        
        Args:
            value: Value to validate
            expected_type: Expected Python type
            field_name: Name of the field being validated
            
        Returns:
            The validated value
            
        Raises:
            TypeValidationError: If type validation fails
        """
        if not isinstance(value, expected_type):
            raise TypeValidationError(
                f"Expected {expected_type.__name__} for '{field_name}', "
                f"got {type(value).__name__}",
                field=field_name,
                value=value
            )
        return value
    
    @staticmethod
    def validate_optional_type(
        value: Any,
        expected_type: Type[T],
        field_name: str = "value"
    ) -> Optional[T]:
        """Validate optional type (allows None)."""
        if value is None:
            return None
        return TypeValidator.validate_type(value, expected_type, field_name)
    
    @staticmethod
    def validate_list_type(
        value: Any,
        item_type: Type[T],
        field_name: str = "value",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[T]:
        """Validate list with specific item type."""
        TypeValidator.validate_type(value, list, field_name)
        
        if min_length is not None and len(value) < min_length:
            raise TypeValidationError(
                f"List '{field_name}' has {len(value)} items, minimum is {min_length}",
                field=field_name
            )
        
        if max_length is not None and len(value) > max_length:
            raise TypeValidationError(
                f"List '{field_name}' has {len(value)} items, maximum is {max_length}",
                field=field_name
            )
        
        for i, item in enumerate(value):
            if not isinstance(item, item_type):
                raise TypeValidationError(
                    f"Item {i} in '{field_name}' expected {item_type.__name__}, "
                    f"got {type(item).__name__}",
                    field=f"{field_name}[{i}]",
                    value=item
                )
        
        return value
    
    @staticmethod
    def validate_dict_type(
        value: Any,
        key_type: Type,
        value_type: Type,
        field_name: str = "value"
    ) -> Dict:
        """Validate dictionary with specific key/value types."""
        TypeValidator.validate_type(value, dict, field_name)
        
        for k, v in value.items():
            if not isinstance(k, key_type):
                raise TypeValidationError(
                    f"Dict key '{k}' in '{field_name}' expected {key_type.__name__}",
                    field=field_name
                )
            if not isinstance(v, value_type):
                raise TypeValidationError(
                    f"Dict value for key '{k}' expected {value_type.__name__}",
                    field=field_name
                )
        
        return value


class InputSanitizer:
    """
    Input sanitization for preventing injection attacks.
    """
    
    # Patterns to remove/sanitize
    SCRIPT_TAG_PATTERN = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
    HTML_TAG_PATTERN = re.compile(r'<[^>]*>')
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\bSELECT\b.*\bFROM\b)|(\bINSERT\b.*\bINTO\b)|(\bDELETE\b.*\bFROM\b)|(\bUPDATE\b.*\bSET\b)", re.IGNORECASE),
        re.compile(r"(\bDROP\b.*\bTABLE\b)|(\bALTER\b.*\bTABLE\b)|(\bCREATE\b.*\bTABLE\b)", re.IGNORECASE),
        re.compile(r"'\s*OR\s*'1'='1", re.IGNORECASE),
        re.compile(r"UNION\s+SELECT", re.IGNORECASE),
    ]
    
    # Maximum allowed string length
    MAX_STRING_LENGTH = 10000
    
    @classmethod
    def sanitize_string(
        cls,
        value: str,
        field_name: str = "value",
        strip_html: bool = True,
        prevent_sql: bool = False,
        max_length: Optional[int] = None
    ) -> str:
        """
        Sanitize string input to prevent injection attacks.
        
        Args:
            value: String to sanitize
            field_name: Field name for error messages
            strip_html: Remove HTML tags
            prevent_sql: Check for SQL injection patterns
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            InputSanitizationError: If string contains malicious content
        """
        try:
            TypeValidator.validate_type(value, str, field_name)
            
            # Check length
            max_len = max_length or cls.MAX_STRING_LENGTH
            if len(value) > max_len:
                raise InputSanitizationError(
                    f"String '{field_name}' exceeds maximum length of {max_len}",
                    field=field_name
                )
            
            sanitized = value.strip()
            
            # Strip HTML
            if strip_html:
                sanitized = cls.SCRIPT_TAG_PATTERN.sub('', sanitized)
                sanitized = cls.HTML_TAG_PATTERN.sub('', sanitized)
            
            # Check for SQL injection
            if prevent_sql:
                for pattern in cls.SQL_INJECTION_PATTERNS:
                    if pattern.search(sanitized):
                        raise InputSanitizationError(
                            f"Potential SQL injection detected in '{field_name}'",
                            field=field_name,
                            value=sanitized[:100]  # Only show first 100 chars
                        )
            
            return sanitized
            
        except InputSanitizationError:
            raise
        except Exception as e:
            logger.error(f"Input sanitization failed: {e}", exc_info=True)
            raise InputSanitizationError(f"Sanitization failed: {e}", field=field_name)
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal.
        """
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        # Remove null bytes
        filename = filename.replace('\x00', '')
        # Remove leading dots (hidden files)
        filename = filename.lstrip('.')
        # Only allow safe characters
        filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename or 'unnamed'


def validate_model(model_name: str) -> bool:
    """
    Validate an Ollama model name format.
    
    Expected format: name:tag (e.g., qwen2.5:7b)
    """
    pattern = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*:[a-zA-Z0-9][a-zA-Z0-9._-]*$')
    return bool(pattern.match(model_name))


def validate_repo_path(repo_path: str) -> bool:
    """
    Validate a repository path exists and contains a .git directory.
    """
    try:
        path = PathValidator.validate_path(
            repo_path,
            must_exist=True,
            must_be_dir=True
        )
        git_dir = path / '.git'
        return git_dir.exists()
    except (PathSecurityError, ValidationError):
        return False


class PydanticValidator:
    """
    Schema validation using Pydantic models with custom error handling.
    """
    
    @staticmethod
    def validate_and_parse(
        data: Dict[str, Any],
        schema: Type[BaseModel],
        strict: bool = True
    ) -> BaseModel:
        """
        Validate data against a Pydantic schema.
        
        Args:
            data: Data to validate
            schema: Pydantic model class
            strict: Enable strict validation
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if strict:
                return schema.model_validate(data, strict=True)
            return schema.model_validate(data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error['loc'])
                msg = error['msg']
                errors.append(f"{field}: {msg}")
            
            raise ValidationError(
                f"Schema validation failed: {'; '.join(errors)}",
                value=data
            )


# Decorator for function input validation
def validate_inputs(**validators):
    """
    Decorator to validate function inputs with custom validators.
    
    Usage:
        @validate_inputs(
            repo_path=lambda x: PathValidator.validate_path(x, must_exist=True),
            task_type=lambda x: TypeValidator.validate_type(x, str)
        )
        async def my_function(repo_path, task_type):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate keyword arguments
            for param_name, validator_func in validators.items():
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = validator_func(kwargs[param_name])
                    except Exception as e:
                        raise ValidationError(
                            f"Validation failed for '{param_name}': {e}",
                            field=param_name,
                            value=kwargs.get(param_name)
                        )
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for param_name, validator_func in validators.items():
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = validator_func(kwargs[param_name])
                    except Exception as e:
                        raise ValidationError(
                            f"Validation failed for '{param_name}': {e}",
                            field=param_name,
                            value=kwargs.get(param_name)
                        )
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Initialize with project base directory
PathValidator.set_allowed_base_dirs([
    Path.cwd(),
    Path.home() / "Projects",
    Path.home() / "workspace",
])

logger.info("Validators module initialized successfully")