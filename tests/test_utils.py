from __future__ import annotations

import json
import logging
import threading
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from pipeline_transcriber.models.alert import AlertSeverity
from pipeline_transcriber.models.config import AlertsConfig, LoggingConfig, RetryConfig
from pipeline_transcriber.utils.alerts import AlertManager
from pipeline_transcriber.utils.logging import setup_logging
from pipeline_transcriber.utils.retry import run_with_retry
from pipeline_transcriber.utils.secret_mask import _mask_recursive, _mask_value, mask_secrets
from pipeline_transcriber.utils.state import JobState


# ---------------------------------------------------------------------------
# secret_mask
# ---------------------------------------------------------------------------


class TestMaskValue:
    def test_mask_hf_token(self) -> None:
        result = _mask_value("token hf_abcdef123456", [])
        assert "hf_abcdef123456" not in result
        assert "***MASKED***" in result

    def test_mask_github_pat(self) -> None:
        result = _mask_value("github_pat_XXXX123", [])
        assert "github_pat_XXXX123" not in result
        assert "***MASKED***" in result

    def test_mask_env_value(self) -> None:
        result = _mask_value("my key is mysecret123 here", ["mysecret123"])
        assert "mysecret123" not in result
        assert "***MASKED***" in result

    def test_mask_recursive_dict(self) -> None:
        result = _mask_recursive({"key": "hf_abc123def"}, [])
        assert result["key"] == "***MASKED***"

    def test_mask_secrets_processor(self) -> None:
        event_dict = {"msg": "token hf_abc123def456"}
        result = mask_secrets(None, None, event_dict)
        assert "hf_abc123def456" not in result["msg"]
        assert "***MASKED***" in result["msg"]


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


class TestJobState:
    def test_job_state_init(self, tmp_path: object) -> None:
        state = JobState("job1", tmp_path / "job1")
        assert state.completed_stages == []
        assert state.status == "pending"
        assert state.current_stage is None

    def test_mark_stage_started(self, tmp_path: object) -> None:
        state = JobState("job1", tmp_path / "job1")
        state.mark_stage_started("DOWNLOAD")
        assert state.current_stage == "DOWNLOAD"
        assert state.status == "running"
        assert state.state_path.exists()

    def test_mark_stage_completed(self, tmp_path: object) -> None:
        state = JobState("job1", tmp_path / "job1")
        state.mark_stage_started("DOWNLOAD")
        state.mark_stage_completed("DOWNLOAD", 1)
        assert "DOWNLOAD" in state.completed_stages
        assert state.stage_attempts["DOWNLOAD"] == 1
        assert state.current_stage is None

    def test_mark_job_finished(self, tmp_path: object) -> None:
        state = JobState("job1", tmp_path / "job1")
        state.mark_stage_started("DOWNLOAD")
        state.mark_job_finished("success")
        assert state.status == "success"

    def test_state_load_existing(self, tmp_path: object) -> None:
        job_dir = tmp_path / "job1"
        state = JobState("job1", job_dir)
        state.mark_stage_started("DOWNLOAD")
        state.mark_stage_completed("DOWNLOAD", 2)
        state.mark_stage_started("ASR")
        state.mark_stage_completed("ASR", 1)

        loaded = JobState.load("job1", job_dir)
        assert loaded.completed_stages == ["DOWNLOAD", "ASR"]
        assert loaded.stage_attempts["DOWNLOAD"] == 2
        assert loaded.stage_attempts["ASR"] == 1
        assert loaded.current_stage is None

    def test_state_load_nonexistent(self, tmp_path: object) -> None:
        job_dir = tmp_path / "missing_job"
        loaded = JobState.load("missing", job_dir)
        assert loaded.completed_stages == []
        assert loaded.status == "pending"
        assert loaded.current_stage is None


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_alert_disabled(self, tmp_path: object) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        config = AlertsConfig(enabled=False, channels=["jsonl"], alerts_file=alerts_file)
        manager = AlertManager(config)
        manager.send(
            job_id="job1",
            stage="DOWNLOAD",
            severity=AlertSeverity.ERROR,
            error_code="DL_FAIL",
            message="download failed",
            attempts_used=3,
            trace_id="trace-1",
        )
        assert not alerts_file.exists()

    def test_alert_jsonl(self, tmp_path: object) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        config = AlertsConfig(enabled=True, channels=["jsonl"], alerts_file=alerts_file)
        manager = AlertManager(config)
        manager.send(
            job_id="job1",
            stage="DOWNLOAD",
            severity=AlertSeverity.ERROR,
            error_code="DL_FAIL",
            message="download failed",
            attempts_used=3,
            trace_id="trace-1",
        )
        lines = alerts_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["job_id"] == "job1"
        assert data["stage"] == "DOWNLOAD"
        assert data["error_code"] == "DL_FAIL"

    def test_alert_stderr(self, tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
        config = AlertsConfig(enabled=True, channels=["stderr"], alerts_file=tmp_path / "unused.jsonl")
        manager = AlertManager(config)
        manager.send(
            job_id="job1",
            stage="ASR",
            severity=AlertSeverity.WARNING,
            error_code="ASR_TIMEOUT",
            message="asr timed out",
            attempts_used=2,
            trace_id="trace-2",
        )
        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["job_id"] == "job1"
        assert data["stage"] == "ASR"
        assert data["error_code"] == "ASR_TIMEOUT"

    def test_alert_jsonl_write_error_is_swallowed(self, tmp_path: object) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        config = AlertsConfig(enabled=True, channels=["jsonl"], alerts_file=alerts_file)
        manager = AlertManager(config)

        with patch("builtins.open", side_effect=OSError("disk full")):
            manager.send(
                job_id="job1",
                stage="DOWNLOAD",
                severity=AlertSeverity.ERROR,
                error_code="DL_FAIL",
                message="download failed",
                attempts_used=3,
                trace_id="trace-1",
            )

        assert not alerts_file.exists()

    def test_concurrent_alert_writes_remain_valid_jsonl(self, tmp_path: object) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        config = AlertsConfig(enabled=True, channels=["jsonl"], alerts_file=alerts_file)
        manager = AlertManager(config)
        real_open = open

        class SlowAppendProxy:
            def __init__(self, path):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, data: str) -> int:
                for ch in data:
                    with real_open(self.path, "a") as fh:
                        fh.write(ch)
                    time.sleep(0.00005)
                return len(data)

        def slow_open(path, mode="r", *args, **kwargs):
            if path == alerts_file and mode == "a":
                return SlowAppendProxy(path)
            return real_open(path, mode, *args, **kwargs)

        def send_alert(idx: int) -> None:
            manager.send(
                job_id=f"job-{idx}",
                stage="DOWNLOAD",
                severity=AlertSeverity.ERROR,
                error_code=f"ERR_{idx}",
                message="download failed",
                attempts_used=1,
                trace_id=f"trace-{idx}",
            )

        with patch("builtins.open", side_effect=slow_open):
            threads = [threading.Thread(target=send_alert, args=(i,)) for i in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        lines = [line for line in alerts_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 10
        parsed = [json.loads(line) for line in lines]
        assert {item["job_id"] for item in parsed} == {f"job-{i}" for i in range(10)}


class TestLoggingSetup:
    def test_log_records_include_thread_field(self, tmp_path: Path) -> None:
        cfg = LoggingConfig(log_dir=tmp_path / "logs", json=True)
        setup_logging(cfg, batch_id="batch-1")

        structlog.get_logger("phase3-test").bind(job_id="job-1", trace_id="t1").info("custom_event")

        batch_log = cfg.log_dir / "batch_batch-1.jsonl"
        entries = [json.loads(line) for line in batch_log.read_text().splitlines() if line.strip()]
        assert entries
        assert "thread" in entries[-1]

    def test_rotation_and_retention(self, tmp_path: Path) -> None:
        cfg = LoggingConfig(
            log_dir=tmp_path / "logs",
            json=True,
            file_rotation_mb=1,
            retention_count=2,
        )
        setup_logging(cfg, batch_id="rotate")

        logger = logging.getLogger("phase3-rotation")
        large_message = "x" * 250000
        for i in range(18):
            logger.info({"event": f"rotate_{i}", "payload": large_message, "job_id": "job-1", "trace_id": "t1"})

        log_files = sorted(cfg.log_dir.glob("batch_rotate.jsonl*"))
        assert len(log_files) <= 3

    def test_setup_logging_preserves_external_handlers(self, tmp_path: Path) -> None:
        cfg = LoggingConfig(log_dir=tmp_path / "logs", json=True)
        root_logger = logging.getLogger()
        external = logging.StreamHandler(StringIO())
        root_logger.addHandler(external)
        try:
            setup_logging(cfg, batch_id="batch-1")
            setup_logging(cfg, batch_id="batch-2")
            assert external in root_logger.handlers
        finally:
            root_logger.removeHandler(external)
            external.close()

    def test_setup_logging_replaces_only_same_batch_handler(self, tmp_path: Path) -> None:
        cfg = LoggingConfig(log_dir=tmp_path / "logs", json=True)
        root_logger = logging.getLogger()

        setup_logging(cfg, batch_id="batch-1")
        setup_logging(cfg, batch_id="batch-1")

        batch_handlers = [
            handler for handler in root_logger.handlers
            if getattr(handler, "_pipeline_transcriber_batch_id", None) == "batch-1"
        ]
        assert len(batch_handlers) == 1

    def test_second_batch_setup_keeps_first_batch_log_active(self, tmp_path: Path) -> None:
        cfg = LoggingConfig(log_dir=tmp_path / "logs", json=True)

        setup_logging(cfg, batch_id="batch-1")
        structlog.get_logger("phase4-logging").bind(batch_id="batch-1", job_id="job-1", trace_id="t1").info("first")

        setup_logging(cfg, batch_id="batch-2")
        structlog.get_logger("phase4-logging").bind(batch_id="batch-1", job_id="job-1", trace_id="t1").info("first_after_second_setup")
        structlog.get_logger("phase4-logging").bind(batch_id="batch-2", job_id="job-2", trace_id="t2").info("second")

        batch1_log = cfg.log_dir / "batch_batch-1.jsonl"
        batch2_log = cfg.log_dir / "batch_batch-2.jsonl"

        batch1_events = [json.loads(line)["event"] for line in batch1_log.read_text().splitlines() if line.strip()]
        batch2_events = [json.loads(line)["event"] for line in batch2_log.read_text().splitlines() if line.strip()]

        assert "first" in batch1_events
        assert "first_after_second_setup" in batch1_events
        assert "second" in batch2_events

    def test_different_batch_handlers_do_not_accumulate_after_cleanup(self, tmp_path: Path) -> None:
        from pipeline_transcriber.utils.logging import cleanup_batch_logger

        cfg = LoggingConfig(log_dir=tmp_path / "logs", json=True)
        root_logger = logging.getLogger()

        cleanup_batch_logger()

        setup_logging(cfg, batch_id="batch-1")
        setup_logging(cfg, batch_id="batch-2")
        setup_logging(cfg, batch_id="batch-3")

        cleanup_batch_logger("batch-1")
        cleanup_batch_logger("batch-2")
        cleanup_batch_logger("batch-3")

        batch_handlers = [
            handler for handler in root_logger.handlers
            if getattr(handler, "_pipeline_transcriber_batch_id", None) is not None
        ]
        assert batch_handlers == []


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    _RETRY_CFG = RetryConfig(max_attempts=3, backoff_schedule=[0, 0, 0])
    _COMMON = dict(
        stage_name="TEST",
        job_id="job1",
        trace_id="trace-1",
    )

    def test_retry_success_first_attempt(self) -> None:
        result, attempts = run_with_retry(
            func=lambda: "ok",
            validate_func=lambda _r: None,
            can_retry_func=lambda _e: True,
            suggest_fallback_func=lambda _a: None,
            retry_config=self._RETRY_CFG,
            **self._COMMON,
        )
        assert result == "ok"
        assert attempts == 1

    def test_retry_success_after_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pipeline_transcriber.utils.retry as retry_mod

        monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)

        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "recovered"

        result, attempts = run_with_retry(
            func=flaky,
            validate_func=lambda _r: None,
            can_retry_func=lambda _e: True,
            suggest_fallback_func=lambda _a: None,
            retry_config=self._RETRY_CFG,
            **self._COMMON,
        )
        assert result == "recovered"
        assert attempts == 3

    def test_retry_all_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pipeline_transcriber.utils.retry as retry_mod

        monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)

        def always_fail() -> None:
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            run_with_retry(
                func=always_fail,
                validate_func=lambda _r: None,
                can_retry_func=lambda _e: True,
                suggest_fallback_func=lambda _a: None,
                retry_config=self._RETRY_CFG,
                **self._COMMON,
            )

    def test_retry_non_retryable(self) -> None:
        call_count = 0

        def fail_once() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("non-retryable")

        with pytest.raises(TypeError, match="non-retryable"):
            run_with_retry(
                func=fail_once,
                validate_func=lambda _r: None,
                can_retry_func=lambda _e: False,
                suggest_fallback_func=lambda _a: None,
                retry_config=self._RETRY_CFG,
                **self._COMMON,
            )
        assert call_count == 1
