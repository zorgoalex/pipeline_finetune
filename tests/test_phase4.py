"""Phase 4 comprehensive tests: retry integration, full pipeline mock, export, E2E smoke."""
from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.orchestrator import Orchestrator
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.stages.export import ExportStage
from pipeline_transcriber.stages.input_validate import InputValidateStage
from pipeline_transcriber.stages.qa import QaStage
from pipeline_transcriber.stages.finalize import FinalizeReportStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path, **overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.app.work_dir = tmp_path / "output"
    cfg.logging.log_dir = tmp_path / "logs"
    cfg.alerts.alerts_file = tmp_path / "alerts.jsonl"
    cfg.alerts.channels = ["jsonl"]  # avoid stderr noise in tests
    cfg.retry.max_attempts = 1
    cfg.retry.backoff_schedule = [0]
    cfg.vad.enabled = False
    cfg.alignment.enabled = False
    cfg.diarization.enabled = False
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def make_job(job_id: str = "test-job-01", **overrides) -> Job:
    defaults = dict(
        job_id=job_id,
        source_type="local_file",
        source="/tmp/test.wav",
        output_formats=["json", "txt", "srt", "vtt"],
    )
    defaults.update(overrides)
    return Job(**defaults)


def _ok_result():
    return StageResult(status=StageStatus.SUCCESS, artifacts=[], metrics={})


def _ok_validation():
    return ValidationResult(ok=True, checks=[])


# ---------------------------------------------------------------------------
# Reusable mock stages
# ---------------------------------------------------------------------------

class MockInputValidateStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.INPUT_VALIDATE

    def run(self, ctx):
        return _ok_result()

    def validate(self, ctx, result):
        return _ok_validation()


class MockDownloadStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.DOWNLOAD

    def run(self, ctx):
        raw_dir = ctx.artifacts_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        media = raw_dir / "source_media.wav"
        media.write_bytes(b"\x00" * 100)
        meta = raw_dir / "source_meta.json"
        meta.write_text(json.dumps({"source": ctx.job.source, "title": "Mock"}))
        ctx.download_output_path = media
        return StageResult(status=StageStatus.SUCCESS, artifacts=[str(media), str(meta)])

    def validate(self, ctx, result):
        return _ok_validation()


class MockAudioPrepareStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.AUDIO_PREPARE

    def run(self, ctx):
        audio_dir = ctx.artifacts_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio = audio_dir / "audio_16k_mono.wav"
        audio.write_bytes(b"\x00" * 100)
        probe = audio_dir / "audio_probe.json"
        probe.write_text(json.dumps({
            "sample_rate": 16000, "channels": 1, "duration_sec": 15.0,
        }))
        ctx.audio_path = audio
        return StageResult(status=StageStatus.SUCCESS, artifacts=[str(audio), str(probe)])

    def validate(self, ctx, result):
        return _ok_validation()


class MockVadStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.VAD_SEGMENTATION

    def run(self, ctx):
        ctx.vad_segments = [{"start": 0.0, "end": 5.0}]
        return _ok_result()

    def validate(self, ctx, result):
        return _ok_validation()


class MockAsrStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.ASR_TRANSCRIPTION

    def run(self, ctx):
        asr_dir = ctx.artifacts_dir / "asr"
        asr_dir.mkdir(parents=True, exist_ok=True)
        segments = [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Hello world"},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "This is a test"},
        ]
        asr_result = {"segments": segments, "language": ctx.job.language}
        (asr_dir / "raw_asr.json").write_text(json.dumps(asr_result))
        (asr_dir / "asr_segments.jsonl").write_text(
            "\n".join(json.dumps(s) for s in segments) + "\n"
        )
        (asr_dir / "asr_report.json").write_text(json.dumps({"num_segments": len(segments)}))
        ctx.asr_result = asr_result
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[str(asr_dir / "raw_asr.json")],
        )

    def validate(self, ctx, result):
        return _ok_validation()


class MockAlignStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.ALIGNMENT

    def run(self, ctx):
        segments = [
            {
                "id": 0, "start": 0.0, "end": 5.0, "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 2.5},
                    {"word": "world", "start": 2.5, "end": 5.0},
                ],
            },
            {
                "id": 1, "start": 5.0, "end": 10.0, "text": "This is a test",
                "words": [
                    {"word": "This", "start": 5.0, "end": 6.0},
                    {"word": "is", "start": 6.0, "end": 7.0},
                    {"word": "a", "start": 7.0, "end": 8.0},
                    {"word": "test", "start": 8.0, "end": 10.0},
                ],
            },
        ]
        ctx.aligned_result = {"segments": segments, "language": ctx.job.language}
        return _ok_result()

    def validate(self, ctx, result):
        return _ok_validation()


class MockDiarizeStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.SPEAKER_DIARIZATION

    def run(self, ctx):
        ctx.diarization_result = {
            "labels": ["SPEAKER_00", "SPEAKER_01"],
            "segments": [
                {"start": 0.0, "end": 5.0, "label": "SPEAKER_00"},
                {"start": 5.0, "end": 10.0, "label": "SPEAKER_01"},
            ],
        }
        return _ok_result()

    def validate(self, ctx, result):
        return _ok_validation()


class MockAssignSpeakersStage(BaseStage):
    @property
    def stage_name(self):
        return StageName.SPEAKER_ASSIGNMENT

    def run(self, ctx):
        base = ctx.aligned_result or ctx.asr_result or {"segments": []}
        segments = []
        for seg in base.get("segments", []):
            seg_copy = dict(seg)
            if seg.get("start", 0) < 5.0:
                seg_copy["speaker"] = "SPEAKER_00"
            else:
                seg_copy["speaker"] = "SPEAKER_01"
            segments.append(seg_copy)
        ctx.fused_result = {"segments": segments, "language": base.get("language", "auto")}
        return _ok_result()

    def validate(self, ctx, result):
        return _ok_validation()


# ---------------------------------------------------------------------------
# Stage sequence builders
# ---------------------------------------------------------------------------

def _build_7_stages(config, job=None):
    """Minimal pipeline: no vad, alignment, diarization."""
    return [
        MockInputValidateStage(),
        MockDownloadStage(),
        MockAudioPrepareStage(),
        MockAsrStage(),
        ExportStage(),
        QaStage(),
        FinalizeReportStage(),
    ]


def _build_all_11_stages(config, job=None):
    """Full pipeline with all optional stages."""
    return [
        MockInputValidateStage(),
        MockDownloadStage(),
        MockAudioPrepareStage(),
        MockVadStage(),
        MockAsrStage(),
        MockAlignStage(),
        MockDiarizeStage(),
        MockAssignSpeakersStage(),
        ExportStage(),
        QaStage(),
        FinalizeReportStage(),
    ]


# ===========================================================================
# 1. Integration Tests - Retry
# ===========================================================================

class TestRetryIntegration:
    """Test the retry mechanism with artificial failures."""

    def test_retry_succeeds_after_failures(self, tmp_path: Path) -> None:
        """Stage fails 3 times then succeeds on 4th attempt."""
        call_count = 0

        class FailThenSucceedStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.ASR_TRANSCRIPTION

            def run(self, ctx):
                nonlocal call_count
                call_count += 1
                if call_count < 4:
                    raise RuntimeError(f"Transient error (attempt {call_count})")
                ctx.asr_result = {"segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "ok"}], "language": "en"}
                return _ok_result()

            def validate(self, ctx, result):
                return _ok_validation()

            def can_retry(self, error, ctx):
                return True

        def build_retry_stages(config, job=None):
            return [
                MockInputValidateStage(),
                MockDownloadStage(),
                MockAudioPrepareStage(),
                FailThenSucceedStage(),
                ExportStage(),
                QaStage(),
                FinalizeReportStage(),
            ]

        cfg = make_config(tmp_path)
        cfg.retry.max_attempts = 5
        cfg.retry.backoff_schedule = [0, 0, 0, 0, 0]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=build_retry_stages):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job()])

        assert report["success"] == 1
        assert report["failed"] == 0

        state_path = Path(cfg.app.work_dir) / "test-job-01" / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert state["stage_attempts"]["ASR_TRANSCRIPTION"] == 4
        assert call_count == 4

    def test_retry_exhausted_creates_alert(self, tmp_path: Path) -> None:
        """Stage always fails -> job fails and alert is written."""

        class AlwaysFailStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.ASR_TRANSCRIPTION

            def run(self, ctx):
                raise RuntimeError("Permanent failure")

            def validate(self, ctx, result):
                return _ok_validation()

            def can_retry(self, error, ctx):
                return True

        def build_fail_stages(config, job=None):
            return [
                MockInputValidateStage(),
                MockDownloadStage(),
                MockAudioPrepareStage(),
                AlwaysFailStage(),
            ]

        cfg = make_config(tmp_path)
        cfg.retry.max_attempts = 3
        cfg.retry.backoff_schedule = [0, 0, 0]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=build_fail_stages):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job()])

        assert report["failed"] == 1
        assert report["success"] == 0

        state_path = Path(cfg.app.work_dir) / "test-job-01" / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "failed"

        alerts_path = cfg.alerts.alerts_file
        assert alerts_path.exists()
        alerts = [json.loads(line) for line in alerts_path.read_text().strip().split("\n") if line.strip()]
        assert len(alerts) >= 1
        assert "ASR_TRANSCRIPTION" in alerts[0].get("stage", "") or "ASR_TRANSCRIPTION" in alerts[0].get("error_code", "")

    def test_non_retryable_fails_immediately(self, tmp_path: Path) -> None:
        """Stage with can_retry=False fails after 1 attempt, not 5."""
        call_count = 0

        class NonRetryableStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.ASR_TRANSCRIPTION

            def run(self, ctx):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("Non-retryable error")

            def validate(self, ctx, result):
                return _ok_validation()

            def can_retry(self, error, ctx):
                return False

        def build_nonretry_stages(config, job=None):
            return [
                MockInputValidateStage(),
                MockDownloadStage(),
                MockAudioPrepareStage(),
                NonRetryableStage(),
            ]

        cfg = make_config(tmp_path)
        cfg.retry.max_attempts = 5
        cfg.retry.backoff_schedule = [0, 0, 0, 0, 0]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=build_nonretry_stages):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job()])

        assert report["failed"] == 1
        assert call_count == 1


# ===========================================================================
# 2. Integration Tests - Full Pipeline Mock
# ===========================================================================

class TestFullPipelineMock:
    """Full pipeline integration with mock stages."""

    @patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_build_all_11_stages)
    def test_full_pipeline_with_diarization(self, mock_stages, tmp_path: Path) -> None:
        """Run with diarization enabled (mock stages), verify all 11 stages complete."""
        cfg = make_config(tmp_path)
        cfg.vad.enabled = True
        cfg.alignment.enabled = True
        cfg.diarization.enabled = True
        job = make_job(enable_diarization=True)

        orch = Orchestrator(cfg)
        report = orch.run_batch([job])

        assert report["success"] == 1

        job_dir = Path(cfg.app.work_dir) / job.job_id
        state_path = job_dir / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert len(state["completed_stages"]) == 11

        feedback_dir = job_dir / "stage_feedback"
        assert feedback_dir.exists()
        feedback_files = list(feedback_dir.glob("*.json"))
        assert len(feedback_files) == 11

    @patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_build_7_stages)
    def test_full_pipeline_without_optional(self, mock_stages, tmp_path: Path) -> None:
        """Run without vad/alignment/diarization -> only 7 stages run."""
        cfg = make_config(tmp_path)
        job = make_job()

        orch = Orchestrator(cfg)
        report = orch.run_batch([job])

        assert report["success"] == 1

        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert len(state["completed_stages"]) == 7

    def test_partial_status_on_optional_failure(self, tmp_path: Path) -> None:
        """VAD stage always fails -> job status is 'partial' (not failed)."""

        class FailingVadStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.VAD_SEGMENTATION

            def run(self, ctx):
                raise RuntimeError("VAD failed")

            def validate(self, ctx, result):
                return _ok_validation()

            def can_retry(self, error, ctx):
                return True

        def build_partial_stages(config, job=None):
            return [
                MockInputValidateStage(),
                MockDownloadStage(),
                MockAudioPrepareStage(),
                FailingVadStage(),
                MockAsrStage(),
                ExportStage(),
                QaStage(),
                FinalizeReportStage(),
            ]

        cfg = make_config(tmp_path)
        cfg.vad.enabled = True
        cfg.retry.max_attempts = 1
        cfg.retry.backoff_schedule = [0]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=build_partial_stages):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job()])

        assert report["partial"] == 1
        assert report["success"] == 0
        assert report["failed"] == 0

        state_path = Path(cfg.app.work_dir) / "test-job-01" / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "partial"


# ===========================================================================
# 3. Export final.json Tests
# ===========================================================================

class TestFinalJsonSchema:
    """Verify final.json content and schema."""

    def _run_export(self, ctx: StageContext) -> dict:
        """Run ExportStage and return parsed final.json."""
        stage = ExportStage()
        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS
        final_path = ctx.job_dir / "final.json"
        assert final_path.exists()
        return json.loads(final_path.read_text())

    def _make_ctx(self, tmp_path: Path, **overrides) -> StageContext:
        cfg = PipelineConfig()
        cfg.app.work_dir = tmp_path / "output"
        job = make_job()
        job_dir = tmp_path / "output" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        ctx = StageContext(
            job=job,
            config=cfg,
            job_dir=job_dir,
            batch_id="test-batch",
            trace_id="test-trace",
        )
        for k, v in overrides.items():
            setattr(ctx, k, v)
        return ctx

    def test_final_json_has_required_fields(self, tmp_path: Path) -> None:
        """Verify final.json contains all required fields per spec."""
        ctx = self._make_ctx(tmp_path, asr_result={
            "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "Hello"}],
            "language": "en",
        })
        data = self._run_export(ctx)

        required_fields = [
            "job_id", "status", "source", "source_type", "language",
            "model", "device", "timings_type", "diarization_enabled",
            "audio", "speakers", "segments", "artifacts", "pipeline",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_final_json_timings_type_word(self, tmp_path: Path) -> None:
        """Aligned result with words -> timings_type='word'."""
        ctx = self._make_ctx(tmp_path, aligned_result={
            "segments": [
                {
                    "id": 0, "start": 0.0, "end": 5.0, "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 2.5},
                        {"word": "world", "start": 2.5, "end": 5.0},
                    ],
                },
            ],
            "language": "en",
        })
        data = self._run_export(ctx)
        assert data["timings_type"] == "word"

    def test_final_json_timings_type_segment(self, tmp_path: Path) -> None:
        """ASR result without words -> timings_type='segment'."""
        ctx = self._make_ctx(tmp_path, asr_result={
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.0, "text": "Hello world"},
            ],
            "language": "en",
        })
        data = self._run_export(ctx)
        assert data["timings_type"] == "segment"

    def test_final_json_speaker_stats(self, tmp_path: Path) -> None:
        """Fused result with 2 speakers -> speakers list has correct stats."""
        ctx = self._make_ctx(tmp_path, fused_result={
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
                {"id": 1, "start": 5.0, "end": 10.0, "text": "World", "speaker": "SPEAKER_01"},
                {"id": 2, "start": 10.0, "end": 15.0, "text": "Again", "speaker": "SPEAKER_00"},
            ],
            "language": "en",
        })
        # Enable diarization in config and job so diarization_enabled=True
        ctx.config.diarization.enabled = True
        ctx.job = make_job(enable_diarization=True)

        data = self._run_export(ctx)
        assert data["diarization_enabled"] is True
        assert len(data["speakers"]) == 2

        spk_map = {s["id"]: s for s in data["speakers"]}
        assert "SPEAKER_00" in spk_map
        assert "SPEAKER_01" in spk_map
        assert spk_map["SPEAKER_00"]["num_segments"] == 2
        assert spk_map["SPEAKER_01"]["num_segments"] == 1
        assert abs(spk_map["SPEAKER_00"]["total_duration"] - 10.0) < 0.01
        assert abs(spk_map["SPEAKER_01"]["total_duration"] - 5.0) < 0.01


@patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_build_all_11_stages)
class TestPhase1DataContracts:
    """Phase 1 data contract tests against the full mocked pipeline."""

    def test_final_json_contains_phase1_contract_fields(self, _mock_stages, tmp_path: Path) -> None:
        cfg = make_config(tmp_path, **{"alignment.enabled": True, "diarization.enabled": True})
        orch = Orchestrator(cfg)
        job = make_job(enable_diarization=True, enable_word_timestamps=True)

        report = orch.run_batch([job])
        assert report["success"] == 1

        job_dir = Path(cfg.app.work_dir) / job.job_id
        final_data = json.loads((job_dir / "final.json").read_text())
        report_data = json.loads((job_dir / "report.json").read_text())
        qa_report = json.loads((job_dir / "qa_report.json").read_text())

        assert final_data["status"] == report_data["status"] == "success"
        assert final_data["qa"] == qa_report
        assert "metrics" in final_data
        assert "processing_time_sec" in final_data["metrics"]
        assert isinstance(final_data["metrics"]["processing_time_sec"], (int, float))
        assert "rtf" in final_data["metrics"]
        assert isinstance(final_data["metrics"]["rtf"], (int, float))
        assert "config_snapshot" in final_data["pipeline"]
        assert final_data["pipeline"]["config_snapshot"]["asr"]["model_name"] == cfg.asr.model_name

    def test_root_segments_and_words_artifacts_are_written_and_advertised(
        self, _mock_stages, tmp_path: Path,
    ) -> None:
        cfg = make_config(tmp_path, **{"alignment.enabled": True, "diarization.enabled": True})
        orch = Orchestrator(cfg)
        job = make_job(enable_diarization=True, enable_word_timestamps=True)

        report = orch.run_batch([job])
        assert report["success"] == 1

        job_dir = Path(cfg.app.work_dir) / job.job_id
        segments_path = job_dir / "segments.jsonl"
        words_path = job_dir / "words.jsonl"
        assert segments_path.exists()
        assert words_path.exists()

        segments_lines = [json.loads(line) for line in segments_path.read_text().splitlines() if line.strip()]
        words_lines = [json.loads(line) for line in words_path.read_text().splitlines() if line.strip()]
        assert len(segments_lines) == 2
        assert len(words_lines) == 6

        final_data = json.loads((job_dir / "final.json").read_text())
        assert final_data["artifacts"]["segments_jsonl"] == "segments.jsonl"
        assert final_data["artifacts"]["words_jsonl"] == "words.jsonl"


# ===========================================================================
# 4. E2E Smoke Test
# ===========================================================================

def _create_wav_file(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> None:
    """Create a real WAV file with silence."""
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))


class TestE2ESmoke:
    """End-to-end smoke test with real temp WAV and mock stages."""

    def test_e2e_local_file_mock_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline with real WAV, mock stages, verify all artifacts."""
        # Create a real WAV file
        wav_path = tmp_path / "input.wav"
        _create_wav_file(wav_path, duration_sec=2.0)
        assert wav_path.exists()

        cfg = make_config(tmp_path)
        cfg.vad.enabled = True
        cfg.alignment.enabled = True
        cfg.diarization.enabled = True
        cfg.retry.max_attempts = 1
        cfg.retry.backoff_schedule = [0]

        job = make_job(
            job_id="e2e-smoke-01",
            source=str(wav_path),
            enable_diarization=True,
        )

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=_build_all_11_stages):
            orch = Orchestrator(cfg)
            report = orch.run_batch([job])

        # Verify batch_report.json
        batch_report_path = Path(cfg.app.work_dir) / "batch_report.json"
        assert batch_report_path.exists()
        batch_report = json.loads(batch_report_path.read_text())
        assert batch_report["success"] == 1

        job_dir = Path(cfg.app.work_dir) / job.job_id

        # Verify final.json
        final_path = job_dir / "final.json"
        assert final_path.exists()
        final_data = json.loads(final_path.read_text())
        assert final_data["job_id"] == job.job_id
        assert final_data["status"] == "success"

        # Verify transcript files
        assert (job_dir / "transcript.txt").exists()
        assert (job_dir / "transcript.srt").exists()
        assert (job_dir / "transcript.vtt").exists()

        # Verify report.json and qa_summary.json
        assert (job_dir / "report.json").exists()
        assert (job_dir / "qa_summary.json").exists()

        # Verify state.json shows success
        state_path = job_dir / "state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"

        # Verify stage_feedback has entries for each stage
        feedback_dir = job_dir / "stage_feedback"
        assert feedback_dir.exists()
        feedback_files = sorted(f.stem for f in feedback_dir.glob("*.json"))
        assert len(feedback_files) == 11

        # All 11 stage names should be present
        expected_stages = {
            "INPUT_VALIDATE", "DOWNLOAD", "AUDIO_PREPARE",
            "VAD_SEGMENTATION", "ASR_TRANSCRIPTION", "ALIGNMENT",
            "SPEAKER_DIARIZATION", "SPEAKER_ASSIGNMENT",
            "EXPORTER", "QA_VALIDATOR", "FINALIZE_REPORT",
        }
        assert set(feedback_files) == expected_stages
