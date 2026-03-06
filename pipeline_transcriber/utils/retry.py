"""Generic retry engine."""
from __future__ import annotations

import time
from typing import Any, Callable

import structlog

from pipeline_transcriber.models.config import RetryConfig
from pipeline_transcriber.models.stage import StageValidationError, ValidationResult

logger = structlog.get_logger(__name__)


def run_with_retry(
    func: Callable[[], Any],
    validate_func: Callable[[Any], ValidationResult | None],
    can_retry_func: Callable[[Exception], bool],
    suggest_fallback_func: Callable[[int], dict[str, Any]],
    retry_config: RetryConfig,
    *,
    stage_name: str,
    job_id: str,
    trace_id: str,
) -> tuple[Any, int]:
    """Execute *func* with retries governed by *retry_config*.

    Returns a tuple of (result, attempts_used).
    Raises the last exception if all attempts are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            result = func()

            # Run validation and attach result
            validation = validate_func(result)
            if validation is not None:
                result.validation = validation
                if not validation.ok:
                    raise StageValidationError(validation)

            return result, attempt
        except Exception as exc:
            last_exception = exc
            log = logger.bind(
                stage=stage_name,
                job_id=job_id,
                trace_id=trace_id,
                attempt=attempt,
                max_attempts=retry_config.max_attempts,
                error=str(exc),
            )

            # Validation failures with retry_recommended=False skip retry
            if (
                isinstance(exc, StageValidationError)
                and exc.validation
                and not exc.validation.retry_recommended
            ):
                log.error("validation_failed_no_retry")
                exc.attempts_used = attempt  # type: ignore[attr-defined]
                raise

            if not can_retry_func(exc):
                log.error("non_retryable_error")
                exc.attempts_used = attempt  # type: ignore[attr-defined]
                raise

            if attempt < retry_config.max_attempts:
                backoff_index = min(attempt - 1, len(retry_config.backoff_schedule) - 1)
                sleep_secs = retry_config.backoff_schedule[backoff_index]
                log.warning("retry_scheduled", backoff_secs=sleep_secs)
                time.sleep(sleep_secs)
                fallback_info = suggest_fallback_func(attempt)
                if fallback_info:
                    log.info("fallback_applied", **fallback_info)
            else:
                log.error("all_attempts_exhausted")

    if last_exception is not None:
        last_exception.attempts_used = retry_config.max_attempts  # type: ignore[attr-defined]
        raise last_exception
    # Dead code: loop always sets last_exception or returns
    raise RuntimeError(
        f"All {retry_config.max_attempts} attempts exhausted for stage "
        f"{stage_name!r} (job={job_id}, trace={trace_id})"
    )
