from __future__ import annotations

import pytest

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
        assert len(stages) == 10  # FinalizeReportStage runs outside sequence

    def test_vad_disabled(self):
        config = PipelineConfig()
        config.vad.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 9
        assert not any(isinstance(s, VadStage) for s in stages)

    def test_diarization_disabled(self):
        config = PipelineConfig()
        config.diarization.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 8
        assert not any(isinstance(s, DiarizeStage) for s in stages)
        assert not any(isinstance(s, AssignSpeakersStage) for s in stages)

    def test_alignment_disabled(self):
        config = PipelineConfig()
        config.alignment.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 9
        assert not any(isinstance(s, AlignStage) for s in stages)

    def test_all_optional_disabled(self):
        config = PipelineConfig()
        config.vad.enabled = False
        config.alignment.enabled = False
        config.diarization.enabled = False
        stages = build_stage_sequence(config)
        assert len(stages) == 6
        stage_names = [s.stage_name for s in stages]
        assert StageName.INPUT_VALIDATE in stage_names
        assert StageName.DOWNLOAD in stage_names
        assert StageName.AUDIO_PREPARE in stage_names
        assert StageName.ASR_TRANSCRIPTION in stage_names
        assert StageName.EXPORTER in stage_names
        assert StageName.QA_VALIDATOR in stage_names
        # FinalizeReportStage runs outside the sequence (guaranteed via try/finally)


# ---------------------------------------------------------------------------
# Stage stub run tests
# ---------------------------------------------------------------------------


class TestStageRuns:
    def test_input_validate_runs(self, tmp_path):
        ctx = make_context(tmp_path)
        stage = InputValidateStage()
        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

    def test_download_creates_artifacts(self, tmp_path):
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
        assert (ctx.job_dir / "transcript.srt").exists()
        assert (ctx.job_dir / "transcript.vtt").exists()

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
    def test_validate_after_run(self, tmp_path):
        ctx = make_context(tmp_path)
        stage = DownloadStage()
        result = stage.run(ctx)
        validation = stage.validate(ctx, result)
        assert validation.ok is True
