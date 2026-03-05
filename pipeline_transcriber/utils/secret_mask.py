"""structlog processor that masks secrets in log output."""
from __future__ import annotations

import os
import re
from typing import Any

_SECRET_PATTERN = re.compile(
    r"(hf_[A-Za-z0-9]+|sk-[A-Za-z0-9]+|ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+)"
)

_SECRET_ENV_VARS: set[str] = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}

_MASK = "***MASKED***"


def _collect_secret_values() -> list[str]:
    """Read secret values from known environment variables."""
    values: list[str] = []
    for var in _SECRET_ENV_VARS:
        val = os.environ.get(var)
        if val:
            values.append(val)
    return values


def _mask_value(value: str, secret_values: list[str]) -> str:
    """Replace exact secret values and regex-matched tokens in a string."""
    for secret in secret_values:
        if secret in value:
            value = value.replace(secret, _MASK)
    value = _SECRET_PATTERN.sub(_MASK, value)
    return value


def _mask_recursive(obj: Any, secret_values: list[str]) -> Any:
    """Recursively mask secrets in nested structures."""
    if isinstance(obj, str):
        return _mask_value(obj, secret_values)
    if isinstance(obj, dict):
        return {k: _mask_recursive(v, secret_values) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        masked = [_mask_recursive(item, secret_values) for item in obj]
        return type(obj)(masked)
    return obj


def mask_secrets(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor that replaces secrets with '***MASKED***'."""
    secret_values = _collect_secret_values()
    return _mask_recursive(event_dict, secret_values)
