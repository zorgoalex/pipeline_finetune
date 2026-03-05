from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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


def make_job(job_id: str = "test-job-01") -> Job:
    return Job(
        job_id=job_id,
        source_type="local_file",
        source="/tmp/test.wav",
        output_formats=["json", "txt"],
    )


def _mock_build_stage_sequence(config):
    """Build mock stage sequence that doesn't need real tools."""
    from pipeline_transcriber.models.stage import StageName
    from pipeline_transcriber.stages.input_validate import InputValidateStage
    from pipeline_transcriber.stages.qa import QaStage
    from pipeline_transcriber.stages.finalize import FinalizeReportStage
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
            return ValidationResult(ok=True, checks=[], next_stage_allowed=True)

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
            return ValidationResult(ok=True, checks=[], next_stage_allowed=True)

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
            return ValidationResult(ok=True, checks=[], next_stage_allowed=True)

    return [
        InputValidateStage(),
        MockDownloadStage(),
        MockAudioPrepareStage(),
        MockAsrStage(),
        ExportStage(),
        QaStage(),
        FinalizeReportStage(),
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
