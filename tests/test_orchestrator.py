from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import os
import pytest

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.models.stage import (
    CheckResult,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.orchestrator import Orchestrator
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.stages.diarize import DiarizeStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.app.work_dir = tmp_path / "output"
    cfg.logging.log_dir = tmp_path / "logs"
    cfg.alerts.alerts_file = tmp_path / "alerts.jsonl"
    cfg.retry.max_attempts = 1  # fast tests
    cfg.retry.backoff_schedule = [0]
    cfg.vad.enabled = False
    cfg.alignment.enabled = False
    cfg.diarization.enabled = False
    return cfg


def make_job(job_id: str = "test-job-01", **kwargs) -> Job:
    defaults = dict(
        job_id=job_id,
        source_type="local_file",
        source="/tmp/test.wav",
        output_formats=["json", "txt"],
        enable_word_timestamps=False,  # alignment disabled in test config
    )
    defaults.update(kwargs)
    return Job(
        **defaults,
    )


def _mock_build_stage_sequence(config, job=None):
    """Build mock stage sequence that doesn't need real tools."""
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.qa import QaStage
    from pipeline_transcriber.stages.export import ExportStage

    class MockDownloadStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.DOWNLOAD

        def run(self, ctx: StageContext) -> StageResult:
            raw_dir = ctx.artifacts_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            media = raw_dir / "source_media.wav"
            media.write_bytes(b"\x00" * 100)
            meta = raw_dir / "source_meta.json"
            meta.write_text(json.dumps({"source": ctx.job.source, "title": "Mock"}))
            ctx.download_output_path = media
            return StageResult(status=StageStatus.SUCCESS, artifacts=[str(media), str(meta)])

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

    class MockAudioPrepareStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.AUDIO_PREPARE

        def run(self, ctx: StageContext) -> StageResult:
            audio_dir = ctx.artifacts_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio = audio_dir / "audio_16k_mono.wav"
            audio.write_bytes(b"\x00" * 100)
            probe = audio_dir / "audio_probe.json"
            probe.write_text(json.dumps({"sample_rate": 16000, "channels": 1, "duration_sec": 15.0}))
            ctx.audio_path = audio
            return StageResult(status=StageStatus.SUCCESS, artifacts=[str(audio), str(probe)])

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

    class MockAsrStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.ASR_TRANSCRIPTION

        def run(self, ctx: StageContext) -> StageResult:
            asr_dir = ctx.artifacts_dir / "asr"
            asr_dir.mkdir(parents=True, exist_ok=True)
            segments = [{"id": 0, "start": 0.0, "end": 5.0, "text": "Mock segment"}]
            asr_result = {"segments": segments, "language": ctx.job.language}
            raw = asr_dir / "raw_asr.json"
            raw.write_text(json.dumps(asr_result))
            jsonl = asr_dir / "asr_segments.jsonl"
            jsonl.write_text(json.dumps(segments[0]) + "\n")
            report = asr_dir / "asr_report.json"
            report.write_text(json.dumps({"num_segments": 1}))
            ctx.asr_result = asr_result
            return StageResult(status=StageStatus.SUCCESS, artifacts=[str(raw), str(jsonl), str(report)])

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

    return [
        InputValidateStage(),
        MockDownloadStage(),
        MockAudioPrepareStage(),
        MockAsrStage(),
        ExportStage(),
        QaStage(),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
class TestOrchestrator:
    """Unit tests for the Orchestrator batch runner."""

    def test_batch_success(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        report = orch.run_batch([make_job()])

        assert report["success"] == 1
        assert report["failed"] == 0

        report_path = Path(cfg.app.work_dir) / "batch_report.json"
        assert report_path.exists()

    def test_batch_two_jobs(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        report = orch.run_batch([make_job("job-a"), make_job("job-b")])

        assert report["total"] == 2
        assert report["success"] == 2

    def test_state_json_created(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        job = make_job()
        orch.run_batch([job])

        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        assert state_path.exists()

        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert len(state["completed_stages"]) > 0

    def test_stage_feedback_created(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        job = make_job()
        orch.run_batch([job])

        feedback_dir = Path(cfg.app.work_dir) / job.job_id / "stage_feedback"
        assert feedback_dir.exists()
        feedback_files = list(feedback_dir.glob("*.json"))
        assert len(feedback_files) > 0

    def test_stage_feedback_success_contract(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        job = make_job()
        orch.run_batch([job])

        feedback_path = Path(cfg.app.work_dir) / job.job_id / "stage_feedback" / "EXPORTER.json"
        feedback = json.loads(feedback_path.read_text())

        assert {"stage", "attempt", "status", "retry_needed", "retry_reason", "checks", "expected", "actual"} <= set(feedback)
        assert feedback["status"] == "pass"
        assert feedback["retry_reason"] is None
        assert feedback["expected"]["validation_ok"] is True
        assert feedback["actual"]["validation_ok"] is True
        assert isinstance(feedback["actual"]["checks"], list)

    def test_finalizer_feedback_uses_same_contract(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        job = make_job()
        orch.run_batch([job])

        feedback_path = Path(cfg.app.work_dir) / job.job_id / "stage_feedback" / "FINALIZE_REPORT.json"
        feedback = json.loads(feedback_path.read_text())

        assert {"expected", "actual", "retry_reason"} <= set(feedback)
        assert feedback["stage"] == "FINALIZE_REPORT"
        assert feedback["retry_reason"] is None

    def test_stage_level_logs_include_contract_fields(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        job = make_job()
        orch.run_batch([job])

        job_log = cfg.logging.log_dir / f"job_{job.job_id}.jsonl"
        lines = [json.loads(line) for line in job_log.read_text().splitlines() if line.strip()]
        stage_events = [
            line for line in lines
            if line.get("event") in {"stage_started", "stage_succeeded", "stage_failed"}
        ]

        assert stage_events
        sample = stage_events[0]
        assert sample["job_id"] == job.job_id
        assert "stage" in sample
        assert "trace_id" in sample
        assert "event" in sample

    def test_resume_skips_completed(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        job = make_job()

        orch1 = Orchestrator(cfg, batch_id="batch-1")
        report1 = orch1.run_batch([job])
        assert report1["success"] == 1

        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state_before = json.loads(state_path.read_text())

        orch2 = Orchestrator(cfg, batch_id="batch-2")
        report2 = orch2.run_batch([job], resume=True)
        assert report2["success"] == 1

        state_after = json.loads(state_path.read_text())
        assert state_after["status"] == "success"
        assert state_after["completed_stages"] == state_before["completed_stages"]

    def test_batch_report_structure(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        report = orch.run_batch([make_job()])

        required_keys = {"batch_id", "total", "success", "failed", "partial", "timestamp", "jobs"}
        assert required_keys.issubset(report.keys())

    def test_resume_disabled_rejects_resume(self, mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        cfg.app.resume_enabled = False
        orch = Orchestrator(cfg)

        with pytest.raises(ValueError, match="resume is disabled"):
            orch.run_batch([make_job()], resume=True)


def _validation_failure_sequence(config, job=None):
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage

    class ValidationFailureStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.ASR_TRANSCRIPTION

        def run(self, ctx: StageContext) -> StageResult:
            return StageResult(status=StageStatus.SUCCESS, artifacts=[], metrics={"num_segments": 0})

        def validate(self, ctx, result):
            return ValidationResult(
                ok=False,
                checks=[
                    CheckResult(
                        name="asr_segments_non_empty",
                        passed=False,
                        details="ASR produced no segments.",
                    )
                ],
                retry_recommended=True,
                retry_reason="ASR output was empty",
            )

        def can_retry(self, error, ctx):
            return False

    return [
        InputValidateStage(),
        ValidationFailureStage(),
        FinalizeReportStage(),
    ]


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_validation_failure_sequence)
def test_stage_feedback_failure_contract_includes_retry_reason(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    orch = Orchestrator(cfg)
    job = make_job()

    report = orch.run_batch([job])
    assert report["failed"] == 1

    feedback_path = Path(cfg.app.work_dir) / job.job_id / "stage_feedback" / "ASR_TRANSCRIPTION.json"
    feedback = json.loads(feedback_path.read_text())

    assert feedback["status"] == "fail"
    assert feedback["retry_needed"] is True
    assert feedback["retry_reason"] == "ASR output was empty"
    assert feedback["expected"]["validation_ok"] is True
    assert feedback["actual"]["validation_ok"] is False
    assert feedback["actual"]["checks"][0]["name"] == "asr_segments_non_empty"
    assert feedback["actual"]["checks"][0]["passed"] is False


def _sleep_stage_sequence(config, job=None):
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage

    class SleepStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.DOWNLOAD

        def run(self, ctx: StageContext) -> StageResult:
            time.sleep(0.3)
            return StageResult(status=StageStatus.SUCCESS, artifacts=[], metrics={})

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

    return [
        InputValidateStage(),
        SleepStage(),
        FinalizeReportStage(),
    ]


def _temp_stage_sequence(config, job=None):
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage

    class TempStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.DOWNLOAD

        def run(self, ctx: StageContext) -> StageResult:
            scratch = ctx.temp_dir / "scratch.txt"
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.write_text("temp")
            return StageResult(status=StageStatus.SUCCESS, artifacts=[str(scratch)], metrics={})

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

        def cleanup_temp(self, ctx: StageContext) -> None:
            marker = ctx.temp_dir / "cleanup_marker.txt"
            marker.write_text("cleanup_called")

    return [
        InputValidateStage(),
        TempStage(),
        FinalizeReportStage(),
    ]


def _failing_temp_stage_sequence(config, job=None):
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage

    class FailingTempStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.DOWNLOAD

        def run(self, ctx: StageContext) -> StageResult:
            scratch = ctx.temp_dir / "scratch.txt"
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.write_text("temp")
            raise RuntimeError("boom")

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

    return [
        InputValidateStage(),
        FailingTempStage(),
        FinalizeReportStage(),
    ]


def _mixed_temp_stage_sequence(config, job=None):
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage

    class MixedTempStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.DOWNLOAD

        def run(self, ctx: StageContext) -> StageResult:
            scratch = ctx.temp_dir / "scratch.txt"
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.write_text("temp")
            if ctx.job.job_id.endswith("fail"):
                raise RuntimeError("boom")
            return StageResult(status=StageStatus.SUCCESS, artifacts=[str(scratch)], metrics={})

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

    return [
        InputValidateStage(),
        MixedTempStage(),
        FinalizeReportStage(),
    ]


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_sleep_stage_sequence)
def test_parallel_batch_execution_is_faster_than_sequential(_mock_stages, tmp_path: Path) -> None:
    jobs = [make_job("job-a"), make_job("job-b")]

    cfg_seq = make_config(tmp_path / "seq")
    cfg_seq.app.max_parallel_jobs = 1
    t0 = time.monotonic()
    Orchestrator(cfg_seq).run_batch(jobs)
    sequential_duration = time.monotonic() - t0

    cfg_par = make_config(tmp_path / "par")
    cfg_par.app.max_parallel_jobs = 2
    t1 = time.monotonic()
    Orchestrator(cfg_par).run_batch(jobs)
    parallel_duration = time.monotonic() - t1

    assert sequential_duration > 0.5
    assert parallel_duration < sequential_duration * 0.8


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_temp_stage_sequence)
def test_cleanup_policy_always_removes_job_tmp_dir(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.cleanup_policy = "always"
    cfg.app.tmp_dir = tmp_path / "tmp-work"
    batch_id = "cleanup-always"

    report = Orchestrator(cfg, batch_id=batch_id).run_batch([make_job()])
    assert report["success"] == 1

    assert not (cfg.app.tmp_dir / batch_id / "test-job-01").exists()


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_failing_temp_stage_sequence)
def test_cleanup_policy_on_success_keeps_tmp_dir_for_failed_job(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.cleanup_policy = "on_success"
    cfg.app.tmp_dir = tmp_path / "tmp-work"
    batch_id = "cleanup-failed"

    report = Orchestrator(cfg, batch_id=batch_id).run_batch([make_job()])
    assert report["failed"] == 1

    scratch = cfg.app.tmp_dir / batch_id / "test-job-01" / "scratch.txt"
    assert scratch.exists()


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_temp_stage_sequence)
def test_cleanup_policy_never_keeps_tmp_dir_after_success(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.cleanup_policy = "never"
    cfg.app.tmp_dir = tmp_path / "tmp-work"
    batch_id = "cleanup-never"

    report = Orchestrator(cfg, batch_id=batch_id).run_batch([make_job()])
    assert report["success"] == 1

    scratch = cfg.app.tmp_dir / batch_id / "test-job-01" / "scratch.txt"
    assert scratch.exists()


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_failing_temp_stage_sequence)
def test_cleanup_policy_always_removes_tmp_dir_after_failure(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.cleanup_policy = "always"
    cfg.app.tmp_dir = tmp_path / "tmp-work"
    batch_id = "cleanup-always-fail"

    report = Orchestrator(cfg, batch_id=batch_id).run_batch([make_job()])
    assert report["failed"] == 1
    assert not (cfg.app.tmp_dir / batch_id / "test-job-01").exists()


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_temp_stage_sequence)
def test_same_job_id_different_batches_do_not_share_temp_dir(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.cleanup_policy = "never"
    cfg.app.tmp_dir = tmp_path / "tmp-work"
    job = make_job("shared-job")

    Orchestrator(cfg, batch_id="batch-a").run_batch([job])
    Orchestrator(cfg, batch_id="batch-b").run_batch([job])

    scratch_a = cfg.app.tmp_dir / "batch-a" / "shared-job" / "scratch.txt"
    scratch_b = cfg.app.tmp_dir / "batch-b" / "shared-job" / "scratch.txt"
    assert scratch_a.exists()
    assert scratch_b.exists()
    assert scratch_a != scratch_b


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_temp_stage_sequence)
def test_cleanup_policy_removes_empty_batch_tmp_dir(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.cleanup_policy = "always"
    cfg.app.tmp_dir = tmp_path / "tmp-work"
    batch_id = "cleanup-parent"

    report = Orchestrator(cfg, batch_id=batch_id).run_batch([make_job("job-a")])
    assert report["success"] == 1
    assert not (cfg.app.tmp_dir / batch_id).exists()


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mixed_temp_stage_sequence)
def test_cleanup_policy_preserves_batch_tmp_dir_when_sibling_job_remains(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.cleanup_policy = "on_success"
    cfg.app.tmp_dir = tmp_path / "tmp-work"
    batch_id = "cleanup-siblings"
    jobs = [make_job("job-ok"), make_job("job-fail")]

    report = Orchestrator(cfg, batch_id=batch_id).run_batch(jobs)
    assert report["success"] == 1
    assert report["failed"] == 1
    assert (cfg.app.tmp_dir / batch_id).exists()
    assert not (cfg.app.tmp_dir / batch_id / "job-ok").exists()
    assert (cfg.app.tmp_dir / batch_id / "job-fail").exists()


def test_fail_fast_batch_marks_unstarted_jobs_explicitly(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.fail_fast_batch = True
    jobs = [make_job("job-a"), make_job("job-b"), make_job("job-c")]

    with patch.object(Orchestrator, "_run_single_job", side_effect=["failed", "success", "success"]):
        report = Orchestrator(cfg, batch_id="failfast").run_batch(jobs)

    assert report["failed"] == 1
    assert report["aborted"] == 2
    assert list(report["jobs"].keys()) == ["job-a", "job-b", "job-c"]
    assert report["jobs"]["job-a"] == "failed"
    assert report["jobs"]["job-b"] == "aborted_before_start"
    assert report["jobs"]["job-c"] == "aborted_before_start"


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
def test_run_batch_does_not_accumulate_batch_handlers(_mock_stages, tmp_path: Path) -> None:
    import logging
    from pipeline_transcriber.utils.logging import cleanup_batch_logger

    cfg = make_config(tmp_path)
    root_logger = logging.getLogger()
    cleanup_batch_logger()

    for batch_id in ("batch-a", "batch-b", "batch-c"):
        report = Orchestrator(cfg, batch_id=batch_id).run_batch([make_job(f"{batch_id}-job")])
        assert report["success"] == 1

    batch_handlers = [
        handler for handler in root_logger.handlers
        if getattr(handler, "_pipeline_transcriber_batch_id", None) is not None
    ]
    assert batch_handlers == []


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
def test_parallel_batch_keeps_job_logs_isolated(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.max_parallel_jobs = 2
    jobs = [make_job("job-a"), make_job("job-b")]

    report = Orchestrator(cfg).run_batch(jobs)
    assert report["success"] == 2

    for job in jobs:
        job_log = cfg.logging.log_dir / f"job_{job.job_id}.jsonl"
        lines = [json.loads(line) for line in job_log.read_text().splitlines() if line.strip()]
        logged_job_ids = {line.get("job_id") for line in lines if "job_id" in line}
        assert logged_job_ids == {job.job_id}


def _diarization_missing_secret_sequence(config, job=None):
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage

    class AudioReadyStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.AUDIO_PREPARE

        def run(self, ctx: StageContext) -> StageResult:
            audio_dir = ctx.artifacts_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio = audio_dir / "audio_16k_mono.wav"
            audio.write_bytes(b"\x00" * 100)
            ctx.audio_path = audio
            return StageResult(status=StageStatus.SUCCESS, artifacts=[str(audio)], metrics={})

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

    return [
        InputValidateStage(),
        AudioReadyStage(),
        DiarizeStage(),
        FinalizeReportStage(),
    ]


def _always_fail_sequence(config, job=None):
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage

    class FailStage(BaseStage):
        @property
        def stage_name(self):
            return StageName.ASR_TRANSCRIPTION

        def run(self, ctx: StageContext) -> StageResult:
            raise RuntimeError("permanent failure")

        def validate(self, ctx, result):
            return ValidationResult(ok=True, checks=[])

        def can_retry(self, error, ctx):
            return False

    return [
        InputValidateStage(),
        FailStage(),
        FinalizeReportStage(),
    ]


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_diarization_missing_secret_sequence)
def test_missing_secret_emits_specific_alert(_mock_stages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    cfg = make_config(tmp_path)
    cfg.diarization.enabled = True
    job = make_job(enable_diarization=True)

    report = Orchestrator(cfg).run_batch([job])
    assert report["failed"] == 1

    alerts = [
        json.loads(line)
        for line in cfg.alerts.alerts_file.read_text().splitlines()
        if line.strip()
    ]
    assert any(alert["error_code"] == "MISSING_SECRET_HF_TOKEN" for alert in alerts)


@patch("pipeline_transcriber.orchestrator.Orchestrator._run_single_job", side_effect=RuntimeError("worker crashed"))
def test_system_error_emits_alert(_mock_run_single_job, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.max_parallel_jobs = 2
    jobs = [make_job("job-a"), make_job("job-b")]

    report = Orchestrator(cfg).run_batch(jobs)
    assert report["failed"] == 2

    alerts = [
        json.loads(line)
        for line in cfg.alerts.alerts_file.read_text().splitlines()
        if line.strip()
    ]
    assert any(alert["error_code"] == "SYSTEM_JOB_EXECUTION_ERROR" for alert in alerts)


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_always_fail_sequence)
def test_repeated_batch_failures_emit_batch_alert(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    jobs = [make_job("job-a"), make_job("job-b"), make_job("job-c")]

    report = Orchestrator(cfg).run_batch(jobs)
    assert report["failed"] == 3

    alerts = [
        json.loads(line)
        for line in cfg.alerts.alerts_file.read_text().splitlines()
        if line.strip()
    ]
    assert any(alert["error_code"] == "BATCH_REPEATED_FAILURES" for alert in alerts)


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_always_fail_sequence)
def test_alert_dispatch_failure_does_not_mask_stage_failure(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)

    with patch("pipeline_transcriber.orchestrator.AlertManager.send", side_effect=OSError("alerts unavailable")):
        report = Orchestrator(cfg).run_batch([make_job()])

    assert report["failed"] == 1

    job_dir = Path(cfg.app.work_dir) / "test-job-01"
    state = json.loads((job_dir / "state.json").read_text())
    report_json = json.loads((job_dir / "report.json").read_text())
    final_json = json.loads((job_dir / "final.json").read_text())

    assert state["status"] == "failed"
    assert state["finalization_status"] == "success"
    assert report_json["status"] == "failed"
    assert final_json["status"] == "failed"


@patch("pipeline_transcriber.orchestrator.Orchestrator._run_single_job", side_effect=RuntimeError("worker crashed"))
def test_parallel_worker_crash_writes_minimal_job_contract(_mock_run_single_job, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.max_parallel_jobs = 2
    jobs = [make_job("job-a"), make_job("job-b")]

    report = Orchestrator(cfg).run_batch(jobs)
    assert report["failed"] == 2

    for job in jobs:
        job_dir = Path(cfg.app.work_dir) / job.job_id
        state = json.loads((job_dir / "state.json").read_text())
        report_json = json.loads((job_dir / "report.json").read_text())
        final_json = json.loads((job_dir / "final.json").read_text())

        assert state["job_id"] == job.job_id
        assert state["status"] == "failed"
        assert state["finalization_status"] == "failed"
        assert state["failed_stages"]["SYSTEM"]["error"] == "worker crashed"
        assert report_json["status"] == "failed"
        assert report_json["finalized"] is False
        assert final_json["status"] == "failed"
        assert final_json["finalized"] is False
        assert final_json["job_id"] == job.job_id


@patch("pipeline_transcriber.orchestrator.Orchestrator._run_single_job", side_effect=RuntimeError("worker crashed"))
def test_parallel_worker_crash_preserves_existing_state_progress(_mock_run_single_job, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.app.max_parallel_jobs = 2
    jobs = [make_job("job-a"), make_job("job-b")]
    job = jobs[0]
    job_dir = Path(cfg.app.work_dir) / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    existing_state = {
        "job_id": job.job_id,
        "completed_stages": ["INPUT_VALIDATE", "DOWNLOAD"],
        "current_stage": None,
        "status": "running",
        "stage_attempts": {"INPUT_VALIDATE": 1, "DOWNLOAD": 1},
        "failed_stages": {},
        "stage_ledger": [
            {"stage_name": "INPUT_VALIDATE", "status": "success", "attempts": 1, "duration_ms": 1},
            {"stage_name": "DOWNLOAD", "status": "success", "attempts": 1, "duration_ms": 2},
        ],
        "finalization_status": None,
        "execution_plan": ["INPUT_VALIDATE", "DOWNLOAD", "FINALIZE_REPORT"],
        "config_hash": "cfg-hash",
        "job_hash": "job-hash",
        "job_snapshot": {"job_id": job.job_id, "source": job.source},
        "updated_at": "2026-03-07T00:00:00+00:00",
    }
    (job_dir / "state.json").write_text(json.dumps(existing_state))

    report = Orchestrator(cfg).run_batch(jobs)
    assert report["failed"] == 2

    state = json.loads((job_dir / "state.json").read_text())
    assert state["completed_stages"] == ["INPUT_VALIDATE", "DOWNLOAD"]
    assert state["config_hash"] == "cfg-hash"
    assert state["job_hash"] == "job-hash"
    assert state["stage_ledger"][0]["stage_name"] == "INPUT_VALIDATE"
    assert state["stage_ledger"][-1]["stage_name"] == "SYSTEM"
    assert state["stage_ledger"][-1]["status"] == "failed"


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
def test_corrupt_state_json_on_resume_falls_back_and_batch_continues(_mock_stages, tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    jobs = [make_job("job-a"), make_job("job-b")]

    orch1 = Orchestrator(cfg, batch_id="first")
    first_report = orch1.run_batch(jobs)
    assert first_report["success"] == 2

    job_a_dir = Path(cfg.app.work_dir) / "job-a"
    (job_a_dir / "state.json").write_text("{corrupt")

    orch2 = Orchestrator(cfg, batch_id="resume")
    resumed_report = orch2.run_batch(jobs, resume=True)
    assert resumed_report["success"] == 2

    restored_state = json.loads((job_a_dir / "state.json").read_text())
    assert restored_state["status"] == "success"
    assert "INPUT_VALIDATE" in restored_state["completed_stages"]


# ---------------------------------------------------------------------------
# Fix 1: Resume hydration tests
# ---------------------------------------------------------------------------


class TestResumeHydration:
    """Tests for _hydrate_completed_stages, cascade invalidation, and artifact loading."""

    def _setup_job_dir(self, tmp_path, job_id="hydrate-test"):
        cfg = make_config(tmp_path)
        job_dir = Path(cfg.app.work_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return cfg, job_dir

    def test_hydrate_download_from_disk(self, tmp_path):
        """_find_download_file should find the media file in raw/."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        media = raw_dir / "source_media.wav"
        media.write_bytes(b"\x00" * 50)
        meta = raw_dir / "source_meta.json"
        meta.write_text("{}")

        result = Orchestrator._find_download_file(raw_dir)
        assert result == media

    def test_find_download_file_no_dir(self, tmp_path):
        """Missing raw dir returns None."""
        assert Orchestrator._find_download_file(tmp_path / "nonexistent") is None

    def test_find_download_file_empty_dir(self, tmp_path):
        """Empty raw dir (only json) returns None."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "meta.json").write_text("{}")
        assert Orchestrator._find_download_file(raw_dir) is None

    def test_find_download_file_multiple_picks_newest(self, tmp_path):
        """With multiple non-JSON files, picks most recent by mtime."""
        import time
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        old = raw_dir / "old.wav"
        old.write_bytes(b"\x00" * 10)
        time.sleep(0.05)
        new = raw_dir / "new.mp4"
        new.write_bytes(b"\x00" * 10)
        result = Orchestrator._find_download_file(raw_dir)
        assert result == new

    def test_load_diarization_result(self, tmp_path):
        """_load_diarization_result reconstructs segments + num_speakers."""
        seg_path = tmp_path / "diarization_segments.json"
        segments = [
            {"start": 0, "end": 1, "speaker": "A"},
            {"start": 1, "end": 2, "speaker": "B"},
            {"start": 2, "end": 3, "speaker": "A"},
        ]
        seg_path.write_text(json.dumps(segments))
        result = Orchestrator._load_diarization_result(seg_path)
        assert result["num_speakers"] == 2
        assert len(result["segments"]) == 3

    def test_load_diarization_result_missing(self, tmp_path):
        assert Orchestrator._load_diarization_result(tmp_path / "nope.json") is None

    @patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
    def test_cascade_invalidation(self, mock_stages, tmp_path):
        """If AUDIO_PREPARE artifact missing, ASR should be cascade-invalidated."""
        from pipeline_transcriber.utils.state import JobState
        cfg = make_config(tmp_path)
        job = make_job()
        job_dir = Path(cfg.app.work_dir) / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # First run to create artifacts
        orch1 = Orchestrator(cfg, batch_id="b1")
        orch1.run_batch([job])

        # Remove audio artifact to trigger cascade
        audio_file = job_dir / "artifacts" / "audio" / "audio_16k_mono.wav"
        if audio_file.exists():
            audio_file.unlink()

        # Resume should cascade-invalidate AUDIO_PREPARE and ASR
        orch2 = Orchestrator(cfg, batch_id="b2")
        report = orch2.run_batch([job], resume=True)
        assert report["success"] == 1  # Should still succeed after re-running

    @patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
    def test_hydration_corrupt_json(self, mock_stages, tmp_path):
        """Corrupt JSON artifact should trigger cascade invalidation, not crash."""
        cfg = make_config(tmp_path)
        job = make_job()
        job_dir = Path(cfg.app.work_dir) / job.job_id

        # First run
        orch1 = Orchestrator(cfg, batch_id="b1")
        orch1.run_batch([job])

        # Corrupt the ASR raw JSON
        asr_file = job_dir / "artifacts" / "asr" / "raw_asr.json"
        if asr_file.exists():
            asr_file.write_text("{{{corrupt")

        # Resume should handle corrupt JSON gracefully
        orch2 = Orchestrator(cfg, batch_id="b2")
        report = orch2.run_batch([job], resume=True)
        assert report["success"] == 1

    @patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
    def test_resume_hydrates_ctx_fields(self, mock_stages, tmp_path):
        """After successful resume, ctx fields should be hydrated from disk."""
        cfg = make_config(tmp_path)
        job = make_job()

        orch = Orchestrator(cfg, batch_id="b1")
        report = orch.run_batch([job])
        assert report["success"] == 1

        # Verify artifacts exist on disk
        job_dir = Path(cfg.app.work_dir) / job.job_id
        assert (job_dir / "artifacts" / "raw").exists()
        assert (job_dir / "artifacts" / "asr" / "raw_asr.json").exists()

    @patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_mock_build_stage_sequence)
    def test_hydration_map_covers_all_stages(self, mock_stages, tmp_path):
        """HYDRATION_MAP should have entries for all data-producing stages."""
        expected = {
            "DOWNLOAD", "AUDIO_PREPARE", "VAD_SEGMENTATION",
            "ASR_TRANSCRIPTION", "ALIGNMENT", "SPEAKER_DIARIZATION",
            "SPEAKER_ASSIGNMENT",
        }
        assert expected == set(Orchestrator._HYDRATION_MAP.keys())
