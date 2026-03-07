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


def _add_process_info(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add host and pid to every log event."""
    event_dict["host"] = socket.gethostname()
    event_dict["pid"] = os.getpid()
    event_dict["thread"] = threading.current_thread().name
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
    # Clear existing handlers
    root_logger.handlers.clear()

    # 1. StreamHandler to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # 2. RotatingFileHandler for batch log
    batch_log_path = log_dir / f"batch_{batch_id}.jsonl"
    batch_handler = RotatingFileHandler(
        batch_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    batch_handler.setFormatter(formatter)
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

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    return handler


def remove_job_logger(handler: logging.Handler) -> None:
    """Remove a previously added job handler from the root logger."""
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()
