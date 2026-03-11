from __future__ import annotations

import pytest
from unittest.mock import patch

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.models.stage import StageName, StageStatus
from pipeline_transcriber.stages import build_stage_sequence
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.stages.input_validate import InputValidateStage
from pipeline_transcriber.stages.download import DownloadStage
from pipeline_transcriber.stages.export import ExportStage
from pipeline_transcriber.stages.vad import VadStage
from pipeline_transcriber.stages.diarize import DiarizeStage
from pipeline_transcriber.stages.assign_speakers import AssignSpeakersStage
from pipeline_transcriber.stages.align import AlignStage


def make_context(tmp_path):
    # Create a real source file so download stage can copy it
    source_file = tmp_path / "test_source.wav"
    source_file.write_bytes(b"RIFF" + b"\x00" * 100)

    job = Job(
        job_id="test-01",
        source_type="local_file",
        source=str(source_file),
        output_formats=["json", "txt"],
    )
    config = PipelineConfig()
    config.app.work_dir = tmp_path / "output"
    job_dir = tmp_path / "output" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        job=job,
        config=config,
        job_dir=job_dir,
        batch_id="test-batch",
        trace_id="test-trace",
    )


# ---------------------------------------------------------------------------
# build_stage_sequence tests
# ---------------------------------------------------------------------------


class TestBuildStageSequence:
    def test_all_stages_enabled(self):
        config = PipelineConfig()
        stages = build_stage_sequence(config)
        assert len(stages) == 11
        assert stages[-1].stage_name == StageName.FINALIZE_REPORT

    def test_vad_disabled(self):
        config = PipelineConfig()
        config.vad.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 10
        assert not any(isinstance(s, VadStage) for s in stages)

    def test_diarization_disabled(self):
        config = PipelineConfig()
        config.diarization.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 9
        assert not any(isinstance(s, DiarizeStage) for s in stages)
        assert not any(isinstance(s, AssignSpeakersStage) for s in stages)

    def test_alignment_disabled(self):
        config = PipelineConfig()
        config.alignment.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 10
        assert not any(isinstance(s, AlignStage) for s in stages)

    def test_all_optional_disabled(self):
        config = PipelineConfig()
        config.vad.enabled = False
        config.alignment.enabled = False
        config.diarization.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 7
        stage_names = [s.stage_name for s in stages]
        assert StageName.INPUT_VALIDATE in stage_names
        assert StageName.DOWNLOAD in stage_names
        assert StageName.AUDIO_PREPARE in stage_names
        assert StageName.ASR_TRANSCRIPTION in stage_names
        assert StageName.EXPORTER in stage_names
        assert StageName.QA_VALIDATOR in stage_names
        assert StageName.FINALIZE_REPORT in stage_names


# ---------------------------------------------------------------------------
# Stage stub run tests
# ---------------------------------------------------------------------------


class TestStageRuns:
    def test_input_validate_runs(self, tmp_path):
        ctx = make_context(tmp_path)
        stage = InputValidateStage()
        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

    @patch("pipeline_transcriber.stages.download.probe_audio")
    def test_download_creates_artifacts(self, mock_probe, tmp_path):
        mock_probe.return_value = {
            "sample_rate": 16000,
            "channels": 1,
            "duration_sec": 1.0,
            "codec": "pcm_s16le",
            "format_name": "wav",
        }
        ctx = make_context(tmp_path)
        stage = DownloadStage()
        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

        raw_dir = ctx.artifacts_dir / "raw"
        assert raw_dir.exists()
        files = list(raw_dir.iterdir())
        assert len(files) > 0
        assert ctx.download_output_path is not None
        assert ctx.download_output_path.exists()

    def test_export_creates_files(self, tmp_path):
        ctx = make_context(tmp_path)
        # Export needs no prior artifacts for the stub (empty segments is fine)
        stage = ExportStage()
        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

        assert (ctx.job_dir / "final.json").exists()
        assert (ctx.job_dir / "transcript.txt").exists()
        # job.output_formats=["json", "txt"], so srt/vtt should NOT be created
        assert not (ctx.job_dir / "transcript.srt").exists()
        assert not (ctx.job_dir / "transcript.vtt").exists()

    def test_all_stubs_implement_interface(self):
        config = PipelineConfig()
        stages = build_stage_sequence(config)
        for stage in stages:
            assert isinstance(stage, BaseStage)
            assert hasattr(stage, "stage_name")
            assert isinstance(stage.stage_name, StageName)
            assert callable(getattr(stage, "run", None))
            assert callable(getattr(stage, "validate", None))


# ---------------------------------------------------------------------------
# Stage validation tests
# ---------------------------------------------------------------------------


class TestStageValidation:
    @patch("pipeline_transcriber.stages.download.probe_audio")
    def test_validate_after_run(self, mock_probe, tmp_path):
        mock_probe.return_value = {
            "sample_rate": 16000,
            "channels": 1,
            "duration_sec": 1.0,
            "codec": "pcm_s16le",
            "format_name": "wav",
        }
        ctx = make_context(tmp_path)
        stage = DownloadStage()
        result = stage.run(ctx)
        validation = stage.validate(ctx, result)
        assert validation.ok is True


# ---------------------------------------------------------------------------
# Fix 4: Strict preflight (InputValidateStage) tests
# ---------------------------------------------------------------------------


class TestPreflightValidation:
    def _make_ctx(self, tmp_path, **job_kwargs):
        source_file = tmp_path / "test.wav"
        source_file.write_bytes(b"RIFF" + b"\x00" * 100)
        defaults = dict(
            job_id="preflight-test",
            source_type="local_file",
            source=str(source_file),
            output_formats=["json"],
        )
        defaults.update(job_kwargs)
        job = Job(**defaults)
        config = PipelineConfig()
        config.app.work_dir = tmp_path / "output"
        job_dir = tmp_path / "output" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return StageContext(job=job, config=config, job_dir=job_dir,
                           batch_id="test", trace_id="test")

    def test_word_timestamps_without_alignment_fails(self, tmp_path):
        ctx = self._make_ctx(tmp_path, enable_word_timestamps=True)
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("word timestamps" in w for w in result.warnings)

    def test_word_timestamps_with_alignment_ok(self, tmp_path):
        ctx = self._make_ctx(tmp_path, enable_word_timestamps=True)
        ctx.config.alignment.enabled = True
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.SUCCESS

    def test_diarization_without_config_fails(self, tmp_path):
        ctx = self._make_ctx(tmp_path, enable_diarization=True,
                             enable_word_timestamps=False)
        ctx.config.diarization.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("diarization" in w for w in result.warnings)

    def test_diarization_with_config_ok(self, tmp_path):
        ctx = self._make_ctx(tmp_path, enable_diarization=True,
                             enable_word_timestamps=False)
        ctx.config.diarization.enabled = True
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.SUCCESS

    def test_invalid_output_format_fails(self, tmp_path):
        ctx = self._make_ctx(tmp_path, output_formats=["json", "mp3"],
                             enable_word_timestamps=False)
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("mp3" in w for w in result.warnings)

    def test_missing_output_formats_fails(self, tmp_path):
        ctx = self._make_ctx(tmp_path, output_formats=[], enable_word_timestamps=False)
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("non-empty" in w for w in result.warnings)

    def test_valid_output_formats_ok(self, tmp_path):
        ctx = self._make_ctx(tmp_path, output_formats=["json", "srt", "vtt", "txt", "csv", "tsv", "rttm"],
                             enable_diarization=True,
                             enable_word_timestamps=False)
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.SUCCESS

    def test_rttm_requires_diarization(self, tmp_path):
        ctx = self._make_ctx(tmp_path, output_formats=["json", "rttm"], enable_diarization=False,
                             enable_word_timestamps=False)
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("rttm" in w for w in result.warnings)

    def test_vad_clips_requires_vad_enabled(self, tmp_path):
        ctx = self._make_ctx(tmp_path, enable_word_timestamps=False)
        ctx.config.alignment.enabled = False
        ctx.config.asr.mode = "vad_clips"
        ctx.config.vad.enabled = False
        ctx.config.vad.export_clips = True
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("vad.enabled" in w for w in result.warnings)

    def test_vad_clips_requires_export_clips(self, tmp_path):
        ctx = self._make_ctx(tmp_path, enable_word_timestamps=False)
        ctx.config.alignment.enabled = False
        ctx.config.asr.mode = "vad_clips"
        ctx.config.vad.enabled = True
        ctx.config.vad.export_clips = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("vad.export_clips" in w for w in result.warnings)

    def test_expected_speakers_min_zero_fails(self, tmp_path):
        from pipeline_transcriber.models.job import ExpectedSpeakers
        ctx = self._make_ctx(tmp_path, enable_word_timestamps=False,
                             expected_speakers=ExpectedSpeakers(min=0, max=5))
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("min" in w for w in result.warnings)

    def test_expected_speakers_max_lt_min_fails(self, tmp_path):
        from pipeline_transcriber.models.job import ExpectedSpeakers
        ctx = self._make_ctx(tmp_path, enable_word_timestamps=False,
                             expected_speakers=ExpectedSpeakers(min=5, max=2))
        ctx.config.alignment.enabled = False
        result = InputValidateStage().run(ctx)
        assert result.status == StageStatus.FAILED
        assert any("max" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Fix 2: Job-level output_formats override (ExportStage) tests
# ---------------------------------------------------------------------------


class TestExportFormatsOverride:
    def _make_ctx(self, tmp_path, output_formats=None):
        source_file = tmp_path / "test.wav"
        source_file.write_bytes(b"RIFF" + b"\x00" * 100)
        job = Job(
            job_id="export-test",
            source_type="local_file",
            source=str(source_file),
            output_formats=output_formats or [],
        )
        config = PipelineConfig()
        config.app.work_dir = tmp_path / "output"
        job_dir = tmp_path / "output" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return StageContext(job=job, config=config, job_dir=job_dir,
                           batch_id="test", trace_id="test")

    def test_job_formats_override_config(self, tmp_path):
        """Job with output_formats=['srt'] should only produce srt + final.json."""
        ctx = self._make_ctx(tmp_path, output_formats=["srt"])
        ExportStage().run(ctx)
        assert (ctx.job_dir / "transcript.srt").exists()
        assert (ctx.job_dir / "final.json").exists()
        assert not (ctx.job_dir / "transcript.txt").exists()
        assert not (ctx.job_dir / "transcript.vtt").exists()

    def test_empty_formats_uses_config_defaults(self, tmp_path):
        """Empty job.output_formats falls back to config defaults."""
        ctx = self._make_ctx(tmp_path, output_formats=[])
        ctx.config.export.formats = ["txt", "vtt"]
        ExportStage().run(ctx)
        assert (ctx.job_dir / "transcript.txt").exists()
        assert (ctx.job_dir / "transcript.vtt").exists()
        assert not (ctx.job_dir / "transcript.srt").exists()

    def test_validate_respects_job_formats(self, tmp_path):
        """Validate should check only the job-requested formats."""
        ctx = self._make_ctx(tmp_path, output_formats=["txt"])
        stage = ExportStage()
        result = stage.run(ctx)
        validation = stage.validate(ctx, result)
        assert validation.ok is True
        check_names = [c.name for c in validation.checks]
        assert "export_format:txt" in check_names
        assert "export_format:srt" not in check_names


# ---------------------------------------------------------------------------
# Fix 3: Job-level expected_speakers override (DiarizeStage) tests
# ---------------------------------------------------------------------------


class TestDiarizeExpectedSpeakers:
    def test_effective_bounds_from_config(self, tmp_path):
        """Without job override, bounds come from config."""
        job = Job(job_id="diar-test", source_type="local_file", source="/tmp/t.wav")
        config = PipelineConfig()
        config.diarization.min_speakers = 2
        config.diarization.max_speakers = 8
        job_dir = tmp_path / "output" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        ctx = StageContext(job=job, config=config, job_dir=job_dir,
                           batch_id="test", trace_id="test")
        stage = DiarizeStage()
        mn, mx = stage._effective_speaker_bounds(ctx)
        assert mn == 2
        assert mx == 8

    def test_effective_bounds_from_job(self, tmp_path):
        """Job expected_speakers overrides config."""
        from pipeline_transcriber.models.job import ExpectedSpeakers
        job = Job(job_id="diar-test", source_type="local_file", source="/tmp/t.wav",
                  expected_speakers=ExpectedSpeakers(min=3, max=5))
        config = PipelineConfig()
        config.diarization.min_speakers = 1
        config.diarization.max_speakers = 20
        job_dir = tmp_path / "output" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        ctx = StageContext(job=job, config=config, job_dir=job_dir,
                           batch_id="test", trace_id="test")
        stage = DiarizeStage()
        mn, mx = stage._effective_speaker_bounds(ctx)
        assert mn == 3
        assert mx == 5

    def test_validate_uses_effective_bounds(self, tmp_path):
        """Validate checks speaker count against effective bounds, not config."""
        from pipeline_transcriber.models.job import ExpectedSpeakers
        from pipeline_transcriber.models.stage import StageResult, StageStatus
        job = Job(job_id="diar-test", source_type="local_file", source="/tmp/t.wav",
                  expected_speakers=ExpectedSpeakers(min=2, max=4))
        config = PipelineConfig()
        config.diarization.min_speakers = 1
        config.diarization.max_speakers = 20
        job_dir = tmp_path / "output" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        ctx = StageContext(job=job, config=config, job_dir=job_dir,
                           batch_id="test", trace_id="test")
        # Simulate diarization output with 3 speakers (within job bounds)
        ctx.diarization_result = {
            "segments": [
                {"start": 0, "end": 1, "speaker": "A"},
                {"start": 1, "end": 2, "speaker": "B"},
                {"start": 2, "end": 3, "speaker": "C"},
            ],
            "num_speakers": 3,
        }
        # Create artifact files so validate's file checks pass
        diar_dir = ctx.artifacts_dir / "diarization"
        diar_dir.mkdir(parents=True, exist_ok=True)
        (diar_dir / "diarization_raw.rttm").write_text("")
        (diar_dir / "diarization_segments.json").write_text("[]")
        (diar_dir / "diarization_report.json").write_text("{}")

        stage = DiarizeStage()
        result = StageResult(status=StageStatus.SUCCESS,
                             artifacts=[str(diar_dir / f) for f in [
                                 "diarization_raw.rttm", "diarization_segments.json",
                                 "diarization_report.json"]])
        validation = stage.validate(ctx, result)
        # 3 speakers is within [2,4] — speaker_count_in_range should pass
        range_check = next(c for c in validation.checks if c.name == "speaker_count_in_range")
        assert range_check.passed is True

    def test_can_retry_false_for_hf_token_error(self):
        """HfTokenError is not retryable."""
        from pipeline_transcriber.stages.diarize import HfTokenError
        stage = DiarizeStage()
        assert stage.can_retry(HfTokenError("missing"), None) is False
        assert stage.can_retry(RuntimeError("other"), None) is True
