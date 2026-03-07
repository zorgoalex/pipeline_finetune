from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pipeline_transcriber.models.alert import Alert, AlertSeverity
from pipeline_transcriber.models.config import (
    LoggingConfig,
    PipelineConfig,
    load_config,
)
from pipeline_transcriber.models.job import Job, load_jobs
from pipeline_transcriber.orchestrator import Orchestrator
from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    StageError,
    ValidationResult,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_config(self) -> None:
        cfg = PipelineConfig()
        assert cfg.asr.model_name == "small"
        assert cfg.retry.max_attempts == 5
        assert cfg.retry.backoff_schedule == [2, 5, 10, 20, 30]
        assert cfg.app.resume_enabled is True
        assert cfg.logging.level == "INFO"
        assert cfg.export.formats == ["json", "srt", "vtt", "txt"]

    def test_load_config(self) -> None:
        cfg = load_config(ROOT / "config" / "config.example.yaml")
        assert cfg.asr.model_name == "small"
        assert cfg.asr.compute_type == "int8"
        assert cfg.diarization.backend == "pyannote"
        assert cfg.app.cleanup_policy == "on_success"
        assert cfg.alerts.enabled is True

    def test_config_json_alias(self) -> None:
        lc = LoggingConfig(json=True)
        assert lc.json_format is True
        lc2 = LoggingConfig(json=False)
        assert lc2.json_format is False

    def test_config_invalid_level(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


class TestJob:
    def test_job_valid(self) -> None:
        job = Job(
            job_id="x",
            source_type="youtube",
            source="https://example.com/video",
            output_formats=["json"],
        )
        assert job.job_id == "x"
        assert job.source_type == "youtube"

    def test_job_defaults(self) -> None:
        job = Job(
            job_id="y",
            source_type="local_file",
            source="/tmp/audio.wav",
        )
        assert job.language == "auto"
        assert job.enable_diarization is False
        assert job.enable_word_timestamps is True
        assert job.output_formats == []
        assert job.expected_speakers is None
        assert job.metadata == {}

    def test_load_jobs(self) -> None:
        jobs = load_jobs(ROOT / "jobs" / "jobs.example.jsonl")
        assert len(jobs) == 2
        assert jobs[0].job_id == "demo-youtube-01"
        assert jobs[1].source_type == "local_file"

    def test_load_jobs_json(self, tmp_path: Path) -> None:
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps([
            {"job_id": "job-json-1", "source_type": "local_file", "source": "/tmp/a.wav"},
            {"job_id": "job-json-2", "source_type": "youtube", "source": "https://example.com/v"},
        ]))

        jobs = load_jobs(jobs_path)
        assert [job.job_id for job in jobs] == ["job-json-1", "job-json-2"]

    def test_load_jobs_json_object_with_jobs_key(self, tmp_path: Path) -> None:
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps({
            "jobs": [
                {"job_id": "job-json-1", "source_type": "local_file", "source": "/tmp/a.wav"},
            ]
        }))

        jobs = load_jobs(jobs_path)
        assert len(jobs) == 1
        assert jobs[0].job_id == "job-json-1"

    def test_load_jobs_yaml(self, tmp_path: Path) -> None:
        jobs_path = tmp_path / "jobs.yaml"
        jobs_path.write_text(yaml.safe_dump({
            "jobs": [
                {"job_id": "job-yaml-1", "source_type": "local_file", "source": "/tmp/a.wav"},
                {"job_id": "job-yaml-2", "source_type": "youtube", "source": "https://example.com/v"},
            ]
        }))

        jobs = load_jobs(jobs_path)
        assert [job.job_id for job in jobs] == ["job-yaml-1", "job-yaml-2"]

    @pytest.mark.parametrize("suffix, writer", [
        (".json", lambda path: path.write_text(json.dumps([
            {"job_id": "dup-1", "source_type": "local_file", "source": "/tmp/a.wav"},
            {"job_id": "dup-1", "source_type": "local_file", "source": "/tmp/b.wav"},
        ]))),
        (".yaml", lambda path: path.write_text(yaml.safe_dump({
            "jobs": [
                {"job_id": "dup-1", "source_type": "local_file", "source": "/tmp/a.wav"},
                {"job_id": "dup-1", "source_type": "local_file", "source": "/tmp/b.wav"},
            ]
        }))),
    ])
    def test_duplicate_job_ids_still_fail_for_json_and_yaml_inputs(
        self, tmp_path: Path, suffix: str, writer,
    ) -> None:
        jobs_path = tmp_path / f"jobs{suffix}"
        writer(jobs_path)
        jobs = load_jobs(jobs_path)

        cfg = PipelineConfig()
        cfg.app.work_dir = tmp_path / "output"
        cfg.logging.log_dir = tmp_path / "logs"
        cfg.alerts.alerts_file = tmp_path / "alerts.jsonl"

        with pytest.raises(ValueError, match="Duplicate job_id"):
            Orchestrator(cfg).run_batch(jobs)

    def test_job_with_expected_speakers(self) -> None:
        job = Job(
            job_id="z",
            source_type="youtube",
            source="https://example.com/v",
            expected_speakers={"min": 1, "max": 5},
        )
        assert job.expected_speakers is not None
        assert job.expected_speakers.min == 1
        assert job.expected_speakers.max == 5


# ---------------------------------------------------------------------------
# StageResult / ValidationResult
# ---------------------------------------------------------------------------


class TestStageResult:
    def test_stage_result_success(self) -> None:
        result = StageResult(status=StageStatus.SUCCESS)
        assert result.status == StageStatus.SUCCESS
        assert result.error is None
        assert result.artifacts == []
        assert result.warnings == []

    def test_stage_result_with_error(self) -> None:
        err = StageError(
            error_type="RuntimeError",
            message="something broke",
            traceback="Traceback ...",
            retryable=False,
        )
        result = StageResult(status=StageStatus.FAILED, error=err)
        assert result.status == StageStatus.FAILED
        assert result.error is not None
        assert result.error.error_type == "RuntimeError"
        assert result.error.retryable is False

    def test_validation_result_ok(self) -> None:
        vr = ValidationResult(
            ok=True,
            checks=[CheckResult(name="x", passed=True)],
        )
        assert vr.ok is True
        assert len(vr.checks) == 1
        assert vr.checks[0].passed is True
        assert vr.retry_recommended is False

    def test_stage_name_enum(self) -> None:
        assert len(StageName) == 11
        assert StageName.INPUT_VALIDATE.value == "INPUT_VALIDATE"
        assert StageName.FINALIZE_REPORT.value == "FINALIZE_REPORT"


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


class TestAlert:
    def test_alert_creation(self) -> None:
        alert = Alert(
            alert_id="a-001",
            job_id="job-1",
            stage="ASR_TRANSCRIPTION",
            severity=AlertSeverity.ERROR,
            error_code="ASR_TIMEOUT",
            message="Transcription timed out",
            attempts_used=3,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            host="worker-1",
            trace_id="trace-abc",
        )
        assert alert.alert_id == "a-001"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.attempts_used == 3

    def test_alert_severity_enum(self) -> None:
        assert len(AlertSeverity) == 3
        members = {s.value for s in AlertSeverity}
        assert members == {"WARNING", "ERROR", "CRITICAL"}
