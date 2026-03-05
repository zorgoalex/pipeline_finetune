from __future__ import annotations

import json

import pytest

from pipeline_transcriber.models.alert import AlertSeverity
from pipeline_transcriber.models.config import AlertsConfig, RetryConfig
from pipeline_transcriber.utils.alerts import AlertManager
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
