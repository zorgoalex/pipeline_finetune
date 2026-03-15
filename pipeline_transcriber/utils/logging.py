"""Structured JSON logging setup using structlog + stdlib logging."""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Any

import structlog

from pipeline_transcriber.models.config import LoggingConfig
from pipeline_transcriber.utils.secret_mask import mask_secrets


class _JobIdFilter(logging.Filter):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "job_id", None) == self.job_id:
            return True
        if isinstance(record.msg, dict):
            return record.msg.get("job_id") == self.job_id
        return False


class _BatchIdFilter(logging.Filter):
    def __init__(self, batch_id: str):
        super().__init__()
        self.batch_id = batch_id

    def filter(self, record: logging.LogRecord) -> bool:
        batch_id = getattr(record, "batch_id", None)
        if batch_id is None and isinstance(record.msg, dict):
            batch_id = record.msg.get("batch_id")
        return batch_id is None or batch_id == self.batch_id


def _add_process_info(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add host, pid, thread, and spec-mandated default fields to every log event."""
    event_dict["host"] = socket.gethostname()
    event_dict["pid"] = os.getpid()
    event_dict["thread"] = threading.current_thread().name
    # Spec section 8.2: mandatory fields with safe defaults
    event_dict.setdefault("event", event_dict.get("event", method_name))
    event_dict.setdefault("duration_ms", None)
    event_dict.setdefault("attempt", None)
    return event_dict


def _extract_exception_fields(
    logger: Any, method_name: str, event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Extract exception info into discrete fields per spec section 8.3."""
    exc_info = event_dict.get("exc_info")
    if exc_info and exc_info is not True:
        if isinstance(exc_info, tuple) and len(exc_info) == 3:
            exc_type, exc_value, exc_tb = exc_info
        elif isinstance(exc_info, BaseException):
            exc_type = type(exc_info)
            exc_value = exc_info
        else:
            return event_dict
        event_dict.setdefault("exception_type", exc_type.__name__ if exc_type else None)
        event_dict.setdefault("exception_message", str(exc_value) if exc_value else None)
        import traceback
        if hasattr(exc_value, "__traceback__") and exc_value.__traceback__:
            event_dict.setdefault(
                "stacktrace",
                "".join(traceback.format_exception(exc_type, exc_value, exc_value.__traceback__)),
            )
    return event_dict


def setup_logging(
    config: LoggingConfig,
    batch_id: str,
    job_id: str | None = None,
) -> None:
    """Configure structlog and stdlib root logger with handlers."""
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = config.file_rotation_mb * 1024 * 1024
    backup_count = config.retention_count

    # Build shared processors list
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_process_info,
        _extract_exception_fields,
        mask_secrets,
    ]

    if config.json_format:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level))

    # Keep non-managed handlers intact; replace only handlers owned by this setup.
    if not any(getattr(h, "_pipeline_transcriber_stream", False) for h in root_logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler._pipeline_transcriber_stream = True
        root_logger.addHandler(stream_handler)

    existing_batch_handlers = [
        handler for handler in list(root_logger.handlers)
        if getattr(handler, "_pipeline_transcriber_batch_id", None) == batch_id
    ]
    for handler in existing_batch_handlers:
        root_logger.removeHandler(handler)
        handler.close()

    batch_log_path = log_dir / f"batch_{batch_id}.jsonl"
    batch_handler = RotatingFileHandler(
        batch_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    batch_handler.setFormatter(formatter)
    batch_handler.addFilter(_BatchIdFilter(batch_id))
    batch_handler._pipeline_transcriber_batch_id = batch_id
    root_logger.addHandler(batch_handler)

    # 3. Optional per-job RotatingFileHandler
    if job_id is not None:
        job_log_path = log_dir / f"job_{job_id}.jsonl"
        job_handler = RotatingFileHandler(
            job_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        job_handler.setFormatter(formatter)
        job_handler._pipeline_transcriber_job_id = job_id
        root_logger.addHandler(job_handler)


def setup_job_logger(config: LoggingConfig, job_id: str) -> logging.Handler:
    """Add a per-job file handler to the root logger and return it."""
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = config.file_rotation_mb * 1024 * 1024
    backup_count = config.retention_count

    if config.json_format:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    job_log_path = log_dir / f"job_{job_id}.jsonl"
    handler = RotatingFileHandler(
        job_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setFormatter(formatter)
    handler.addFilter(_JobIdFilter(job_id))
    handler._pipeline_transcriber_job_id = job_id

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    return handler


def remove_job_logger(handler: logging.Handler) -> None:
    """Remove a previously added job handler from the root logger."""
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def cleanup_batch_logger(batch_id: str | None = None) -> None:
    """Remove managed batch handlers from the root logger.

    If ``batch_id`` is provided, only handlers for that batch are removed.
    Otherwise all managed batch handlers are removed.
    """
    root_logger = logging.getLogger()
    handlers_to_remove = [
        handler for handler in list(root_logger.handlers)
        if getattr(handler, "_pipeline_transcriber_batch_id", None) is not None
        and (batch_id is None or getattr(handler, "_pipeline_transcriber_batch_id", None) == batch_id)
    ]
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
        handler.close()
