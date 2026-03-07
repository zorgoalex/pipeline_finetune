"""Tests for critical fixes P1-P5 from review + P0 round 2."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_transcriber.models.config import PipelineConfig, RetryConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageEntry,
    StageResult,
    StageStatus,
    StageValidationError,
    ValidationResult,
    compute_job_status,
)
from pipeline_transcriber.orchestrator import Orchestrator
from pipeline_transcriber.stages import build_stage_sequence
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.stages.finalize import FinalizeReportStage
from pipeline_transcriber.utils.retry import run_with_retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.app.work_dir = tmp_path / "output"
    cfg.logging.log_dir = tmp_path / "logs"
    cfg.alerts.alerts_file = tmp_path / "alerts.jsonl"
    cfg.retry.max_attempts = 1
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
    )
    defaults.update(kwargs)
    return Job(**defaults)


# ---------------------------------------------------------------------------
# P1: Validation-driven retry
# ---------------------------------------------------------------------------

class TestValidationDrivenRetry:
    """P1: StageValidationError + retry_recommended drives retry logic."""

    def test_validation_error_has_failed_checks(self):
        validation = ValidationResult(
            ok=False,
            checks=[
                CheckResult(name="check_a", passed=True),
                CheckResult(name="check_b", passed=False, details="bad"),
            ],
        )
        err = StageValidationError(validation)
        assert "check_b" in str(err)
        assert err.validation is validation

    def test_retry_skipped_when_retry_recommended_false(self):
        """When validation fails with retry_recommended=False, retry is skipped."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            return StageResult(status=StageStatus.SUCCESS, artifacts=[])

        def validate_func(result):
            return ValidationResult(
                ok=False,
                checks=[CheckResult(name="bad", passed=False)],
                retry_recommended=False,
            )

        retry_cfg = RetryConfig(max_attempts=5, backoff_schedule=[0, 0, 0, 0, 0])
        with pytest.raises(StageValidationError):
            run_with_retry(
                func=failing_func,
                validate_func=validate_func,
                can_retry_func=lambda e: True,
                suggest_fallback_func=lambda a: {},
                retry_config=retry_cfg,
                stage_name="test",
                job_id="j1",
                trace_id="t1",
            )
        assert call_count == 1  # No retry happened

    def test_retry_happens_when_retry_recommended_true(self):
        """When validation fails with retry_recommended=True and can_retry=True, retries occur."""
        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            return StageResult(status=StageStatus.SUCCESS, artifacts=[])

        def validate_func(result):
            if call_count < 3:
                return ValidationResult(
                    ok=False,
                    checks=[CheckResult(name="bad", passed=False)],
                    retry_recommended=True,
                )
            return ValidationResult(ok=True, checks=[])

        retry_cfg = RetryConfig(max_attempts=5, backoff_schedule=[0, 0, 0, 0, 0])
        result, attempts = run_with_retry(
            func=func,
            validate_func=validate_func,
            can_retry_func=lambda e: True,
            suggest_fallback_func=lambda a: {},
            retry_config=retry_cfg,
            stage_name="test",
            job_id="j1",
            trace_id="t1",
        )
        assert attempts == 3
        assert call_count == 3

    def test_validation_result_attached_to_stage_result(self):
        """Successful validation is attached to result."""
        def func():
            return StageResult(status=StageStatus.SUCCESS, artifacts=[])

        validation = ValidationResult(ok=True, checks=[
            CheckResult(name="ok_check", passed=True),
        ])

        result, _ = run_with_retry(
            func=func,
            validate_func=lambda r: validation,
            can_retry_func=lambda e: False,
            suggest_fallback_func=lambda a: {},
            retry_config=RetryConfig(max_attempts=1, backoff_schedule=[0]),
            stage_name="test",
            job_id="j1",
            trace_id="t1",
        )
        assert result.validation is validation


# ---------------------------------------------------------------------------
# P2: QA gatekeeper - tested in test_phase3.py already, adding edge case
# ---------------------------------------------------------------------------

class TestQaGatekeeper:
    """P2: QA validate() distinguishes hard vs soft failures."""

    def test_soft_failure_passes_validation(self):
        """Non-critical check failure (word_alignment without flag) passes."""
        from pipeline_transcriber.stages.qa import QaStage, QA_METRICS_ALL_PASSED, QA_METRICS_CHECKS

        stage = QaStage()
        job = make_job()
        cfg = make_config(Path("/tmp"))
        cfg.qa.fail_on_missing_word_timestamps = False

        ctx = StageContext(
            job=job, config=cfg, job_dir=Path("/tmp/qa_test"),
            batch_id="b1", trace_id="t1",
        )

        result = StageResult(
            status=StageStatus.SUCCESS,
            metrics={
                QA_METRICS_ALL_PASSED: False,
                QA_METRICS_CHECKS: [
                    {"name": "final_json_exists", "passed": True},
                    {"name": "segments_non_empty", "passed": True},
                    {"name": "time_intervals_valid", "passed": True},
                    {"name": "no_negative_durations", "passed": True},
                    {"name": "word_alignment_ratio", "passed": False, "details": "low ratio"},
                ],
            },
        )

        validation = stage.validate(ctx, result)
        assert validation.ok is True  # soft failure, passes

    def test_hard_failure_fails_validation(self):
        """Hard failure (segments_non_empty) causes validation fail."""
        from pipeline_transcriber.stages.qa import QaStage, QA_METRICS_ALL_PASSED, QA_METRICS_CHECKS

        stage = QaStage()
        job = make_job()
        cfg = make_config(Path("/tmp"))

        ctx = StageContext(
            job=job, config=cfg, job_dir=Path("/tmp/qa_test"),
            batch_id="b1", trace_id="t1",
        )

        result = StageResult(
            status=StageStatus.SUCCESS,
            metrics={
                QA_METRICS_ALL_PASSED: False,
                QA_METRICS_CHECKS: [
                    {"name": "final_json_exists", "passed": True},
                    {"name": "segments_non_empty", "passed": False, "details": "0 segments"},
                    {"name": "time_intervals_valid", "passed": True},
                    {"name": "no_negative_durations", "passed": True},
                ],
            },
        )

        validation = stage.validate(ctx, result)
        assert validation.ok is False
        assert validation.retry_recommended is False


# ---------------------------------------------------------------------------
# P3: Finalize incorporates QA + patches final.json
# ---------------------------------------------------------------------------

class TestFinalizePatching:
    """P3/P5.3: Finalize patches final.json with honest status."""

    def test_finalize_patches_final_json_success(self, tmp_path: Path):
        """When all stages succeed, final.json status = 'success'."""
        job = make_job()
        cfg = make_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        # Create final.json as export would
        final_data = {"job_id": job.job_id, "status": "success", "segments": []}
        (job_dir / "final.json").write_text(json.dumps(final_data))

        ctx = StageContext(
            job=job, config=cfg, job_dir=job_dir,
            batch_id="b1", trace_id="t1",
        )
        # Add a successful stage output
        ctx.stage_outputs["INPUT_VALIDATE"] = StageResult(
            status=StageStatus.SUCCESS, artifacts=[]
        )

        stage = FinalizeReportStage()
        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

        patched = json.loads((job_dir / "final.json").read_text())
        assert patched["status"] == "success"

    def test_finalize_patches_final_json_partial(self, tmp_path: Path):
        """When job_status is 'partial', final.json status = 'partial'."""
        job = make_job()
        cfg = make_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        final_data = {"job_id": job.job_id, "status": "success", "segments": []}
        (job_dir / "final.json").write_text(json.dumps(final_data))

        ctx = StageContext(
            job=job, config=cfg, job_dir=job_dir,
            batch_id="b1", trace_id="t1",
        )

        stage = FinalizeReportStage()
        result = stage.run(ctx, job_status="partial")

        patched = json.loads((job_dir / "final.json").read_text())
        assert patched["status"] == "partial"

    def test_finalize_incorporates_qa_report(self, tmp_path: Path):
        """Finalize includes qa_report.json in report.json."""
        job = make_job()
        cfg = make_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        qa_data = {"all_passed": True, "checks": []}
        (job_dir / "qa_report.json").write_text(json.dumps(qa_data))

        ctx = StageContext(
            job=job, config=cfg, job_dir=job_dir,
            batch_id="b1", trace_id="t1",
        )
        ctx.stage_outputs["ASR"] = StageResult(
            status=StageStatus.SUCCESS, artifacts=[]
        )

        stage = FinalizeReportStage()
        stage.run(ctx)

        report = json.loads((job_dir / "report.json").read_text())
        assert "qa" in report
        assert report["qa"]["all_passed"] is True


# ---------------------------------------------------------------------------
# P4: Config isolation between batch jobs
# ---------------------------------------------------------------------------

class TestConfigIsolation:
    """P4: model_copy(deep=True) prevents config mutation leak."""

    def test_fallback_mutation_does_not_leak_between_jobs(self, tmp_path: Path):
        """Config changes in one job don't affect subsequent jobs or global config."""

        configs_at_start: list[str] = []

        class RecordAndMutateStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.ASR_TRANSCRIPTION

            def run(self, ctx: StageContext) -> StageResult:
                # Record model name BEFORE mutation
                configs_at_start.append(ctx.config.asr.model_name)
                # Mutate config like a fallback would
                ctx.config.asr.model_name = "mutated-model"
                return StageResult(status=StageStatus.SUCCESS, artifacts=[])

            def validate(self, ctx, result):
                return ValidationResult(ok=True, checks=[])

        def build_stages(config, job=None):
            return [RecordAndMutateStage(), FinalizeReportStage()]

        cfg = make_config(tmp_path)
        original_model = cfg.asr.model_name

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", side_effect=build_stages):
            orch = Orchestrator(cfg)
            orch.run_batch([make_job("job-a"), make_job("job-b")])

        # Both jobs should start with original model (isolation works)
        assert configs_at_start[0] == original_model
        assert configs_at_start[1] == original_model
        # Global config unchanged
        assert cfg.asr.model_name == original_model


# ---------------------------------------------------------------------------
# P5: Per-job stage selection and _is_optional_stage
# ---------------------------------------------------------------------------

class TestPerJobStages:
    """P5: build_stage_sequence respects job flags; _is_optional_stage respects them too."""

    def test_alignment_included_when_job_requests_word_timestamps(self):
        cfg = PipelineConfig()
        cfg.alignment.enabled = True
        job = make_job(enable_word_timestamps=True)
        stages = build_stage_sequence(cfg, job)
        stage_names = [s.stage_name for s in stages]
        assert StageName.ALIGNMENT in stage_names

    def test_alignment_excluded_when_job_disables_word_timestamps(self):
        cfg = PipelineConfig()
        cfg.alignment.enabled = True
        job = make_job(enable_word_timestamps=False)
        stages = build_stage_sequence(cfg, job)
        stage_names = [s.stage_name for s in stages]
        assert StageName.ALIGNMENT not in stage_names

    def test_diarization_included_when_job_requests_it(self):
        cfg = PipelineConfig()
        cfg.diarization.enabled = True
        job = make_job(enable_diarization=True)
        stages = build_stage_sequence(cfg, job)
        stage_names = [s.stage_name for s in stages]
        assert StageName.SPEAKER_DIARIZATION in stage_names
        assert StageName.SPEAKER_ASSIGNMENT in stage_names

    def test_diarization_excluded_when_job_disables_it(self):
        cfg = PipelineConfig()
        cfg.diarization.enabled = True
        job = make_job(enable_diarization=False)
        stages = build_stage_sequence(cfg, job)
        stage_names = [s.stage_name for s in stages]
        assert StageName.SPEAKER_DIARIZATION not in stage_names

    def test_backward_compat_no_job(self):
        """build_stage_sequence works without job param (backward compat)."""
        cfg = PipelineConfig()
        cfg.alignment.enabled = True
        cfg.diarization.enabled = True
        stages = build_stage_sequence(cfg)
        stage_names = [s.stage_name for s in stages]
        assert StageName.ALIGNMENT in stage_names
        assert StageName.SPEAKER_DIARIZATION in stage_names


class TestIsOptionalStage:
    """P5.4: _is_optional_stage respects job-level flags."""

    def _get_orchestrator(self, tmp_path: Path) -> Orchestrator:
        return Orchestrator(make_config(tmp_path))

    def test_vad_always_optional(self, tmp_path: Path):
        from pipeline_transcriber.stages.vad import VadStage
        orch = self._get_orchestrator(tmp_path)
        job = make_job(enable_diarization=True, enable_word_timestamps=True)
        assert orch._is_optional_stage(VadStage(), job) is True

    def test_alignment_not_optional_when_word_timestamps_requested(self, tmp_path: Path):
        from pipeline_transcriber.stages.align import AlignStage
        orch = self._get_orchestrator(tmp_path)
        job = make_job(enable_word_timestamps=True)
        assert orch._is_optional_stage(AlignStage(), job) is False

    def test_alignment_optional_when_word_timestamps_not_requested(self, tmp_path: Path):
        from pipeline_transcriber.stages.align import AlignStage
        orch = self._get_orchestrator(tmp_path)
        job = make_job(enable_word_timestamps=False)
        assert orch._is_optional_stage(AlignStage(), job) is True

    def test_diarization_not_optional_when_requested(self, tmp_path: Path):
        from pipeline_transcriber.stages.diarize import DiarizeStage
        orch = self._get_orchestrator(tmp_path)
        job = make_job(enable_diarization=True)
        assert orch._is_optional_stage(DiarizeStage(), job) is False

    def test_diarization_optional_when_not_requested(self, tmp_path: Path):
        from pipeline_transcriber.stages.diarize import DiarizeStage
        orch = self._get_orchestrator(tmp_path)
        job = make_job(enable_diarization=False)
        assert orch._is_optional_stage(DiarizeStage(), job) is True

    def test_core_stage_never_optional(self, tmp_path: Path):
        from pipeline_transcriber.stages.asr import AsrStage
        orch = self._get_orchestrator(tmp_path)
        job = make_job()
        assert orch._is_optional_stage(AsrStage(), job) is False


# ===========================================================================
# Round 2 P0 Fixes
# ===========================================================================


# ---------------------------------------------------------------------------
# Fix 1: compute_job_status from ledger
# ---------------------------------------------------------------------------

class TestComputeJobStatus:
    """compute_job_status derives status from stage ledger."""

    def test_all_success(self):
        ledger = [
            StageEntry(stage_name="A", status="success"),
            StageEntry(stage_name="B", status="success"),
        ]
        assert compute_job_status(ledger, lambda _: False) == "success"

    def test_required_failure_is_failed(self):
        ledger = [
            StageEntry(stage_name="A", status="success"),
            StageEntry(stage_name="B", status="failed"),
        ]
        assert compute_job_status(ledger, lambda _: False) == "failed"

    def test_optional_failure_is_partial(self):
        ledger = [
            StageEntry(stage_name="A", status="success"),
            StageEntry(stage_name="VAD", status="failed"),
        ]
        assert compute_job_status(ledger, lambda n: n == "VAD") == "partial"

    def test_skipped_stages_not_failures(self):
        ledger = [
            StageEntry(stage_name="A", status="skipped", skip_reason="resumed"),
            StageEntry(stage_name="B", status="success"),
        ]
        assert compute_job_status(ledger, lambda _: False) == "success"

    def test_empty_ledger_is_success(self):
        assert compute_job_status([], lambda _: False) == "success"


# ---------------------------------------------------------------------------
# Fix 1: Ledger recorded in orchestrator
# ---------------------------------------------------------------------------

class TestLedgerRecording:
    """Orchestrator records StageEntry items in ctx.stage_ledger."""

    def _make_success_stage(self, name: str):
        class _Stage(BaseStage):
            @property
            def stage_name(self) -> StageName:
                return StageName(name)
            def run(self, ctx: StageContext) -> StageResult:
                return StageResult(status=StageStatus.SUCCESS, artifacts=["a.txt"])
            def validate(self, ctx, result):
                return ValidationResult(ok=True)
        return _Stage()

    def _make_fail_stage(self, name: str):
        class _Stage(BaseStage):
            @property
            def stage_name(self) -> StageName:
                return StageName(name)
            def run(self, ctx: StageContext) -> StageResult:
                raise RuntimeError("boom")
            def validate(self, ctx, result):
                return ValidationResult(ok=True)
        return _Stage()

    def test_success_stages_in_ledger(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        stages = [self._make_success_stage("INPUT_VALIDATE")]

        def _mock_build(config, job=None):
            return stages

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _mock_build):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job()])

        assert report["success"] == 1
        # Check state.json has ledger
        state_path = cfg.app.work_dir / "test-job-01" / "state.json"
        state = json.loads(state_path.read_text())
        assert len(state["stage_ledger"]) == 2
        assert state["stage_ledger"][0]["stage_name"] == "INPUT_VALIDATE"
        assert state["stage_ledger"][0]["status"] == "success"
        assert state["stage_ledger"][1]["stage_name"] == "FINALIZE_REPORT"
        assert state["stage_ledger"][1]["status"] == "success"

    def test_failed_stage_in_ledger(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        stages = [self._make_fail_stage("INPUT_VALIDATE")]

        def _mock_build(config, job=None):
            return stages

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _mock_build):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job()])

        assert report["failed"] == 1
        state_path = cfg.app.work_dir / "test-job-01" / "state.json"
        state = json.loads(state_path.read_text())
        assert len(state["stage_ledger"]) == 2
        assert state["stage_ledger"][0]["status"] == "failed"
        assert "boom" in state["stage_ledger"][0]["error"]
        assert state["stage_ledger"][1]["stage_name"] == "FINALIZE_REPORT"
        assert state["stage_ledger"][1]["status"] == "success"


# ---------------------------------------------------------------------------
# Fix 2: Guaranteed finalization
# ---------------------------------------------------------------------------

class TestGuaranteedFinalization:
    """FinalizeReportStage always runs, even on failure."""

    def test_report_created_on_failure(self, tmp_path: Path):
        """report.json and final.json exist even when a core stage fails."""
        cfg = make_config(tmp_path)

        class FailStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.INPUT_VALIDATE
            def run(self, ctx):
                raise RuntimeError("crash")
            def validate(self, ctx, result):
                return ValidationResult(ok=True)

        def _mock_build(config, job=None):
            return [FailStage()]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _mock_build):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job()])

        assert report["failed"] == 1
        job_dir = cfg.app.work_dir / "test-job-01"
        assert (job_dir / "report.json").exists()
        assert (job_dir / "final.json").exists()
        final = json.loads((job_dir / "final.json").read_text())
        assert final["status"] == "failed"

    def test_finalize_creates_minimal_final_json(self, tmp_path: Path):
        """When export never ran, finalize creates minimal final.json."""
        job = make_job()
        cfg = make_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        ctx = StageContext(
            job=job, config=cfg, job_dir=job_dir,
            batch_id="b1", trace_id="t1",
        )
        # No final.json exists (export never ran)
        stage = FinalizeReportStage()
        stage.run(ctx, job_status="failed")

        final = json.loads((job_dir / "final.json").read_text())
        assert final["status"] == "failed"
        assert final["job_id"] == job.job_id

    def test_finalize_uses_ledger_not_stage_outputs(self, tmp_path: Path):
        """Report uses stage_ledger for stage summaries."""
        job = make_job()
        cfg = make_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        ctx = StageContext(
            job=job, config=cfg, job_dir=job_dir,
            batch_id="b1", trace_id="t1",
        )
        ctx.stage_ledger.append(StageEntry(
            stage_name="ASR_TRANSCRIPTION", status="success",
            attempts=2, duration_ms=1500,
        ))

        stage = FinalizeReportStage()
        stage.run(ctx, job_status="success")

        report = json.loads((job_dir / "report.json").read_text())
        assert len(report["stages"]) == 1
        assert report["stages"][0]["stage"] == "ASR_TRANSCRIPTION"
        assert report["stages"][0]["attempts"] == 2


# ---------------------------------------------------------------------------
# Fix 3: Duplicate job_id validation
# ---------------------------------------------------------------------------

class TestDuplicateJobId:
    """Batch rejects duplicate job_ids."""

    def test_duplicate_job_ids_raises(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        jobs = [make_job("dup"), make_job("dup")]
        with pytest.raises(ValueError, match="Duplicate job_id"):
            orch.run_batch(jobs)

    def test_unique_job_ids_ok(self, tmp_path: Path):
        cfg = make_config(tmp_path)

        class OkStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.INPUT_VALIDATE
            def run(self, ctx):
                return StageResult(status=StageStatus.SUCCESS)
            def validate(self, ctx, result):
                return ValidationResult(ok=True)

        def _mock_build(config, job=None):
            return [OkStage()]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _mock_build):
            orch = Orchestrator(cfg)
            report = orch.run_batch([make_job("a"), make_job("b")])
        assert report["total"] == 2


# ---------------------------------------------------------------------------
# Fix 4: final.json always mandatory in export
# ---------------------------------------------------------------------------

class TestFinalJsonMandatory:
    """ExportStage always writes final.json."""

    def test_final_json_written_without_json_format(self, tmp_path: Path):
        """final.json is created even when formats=['txt']."""
        from pipeline_transcriber.stages.export import ExportStage

        job = make_job()
        cfg = make_config(tmp_path)
        cfg.export.formats = ["txt"]  # no "json"
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "artifacts").mkdir(parents=True)

        ctx = StageContext(
            job=job, config=cfg, job_dir=job_dir,
            batch_id="b1", trace_id="t1",
        )
        ctx.asr_result = {"segments": [{"text": "hello", "start": 0, "end": 1}]}

        stage = ExportStage()
        result = stage.run(ctx)

        assert (job_dir / "final.json").exists()
        assert (job_dir / "transcript.txt").exists()

    def test_dynamic_artifacts_map(self, tmp_path: Path):
        """Artifacts map only includes files that exist on disk."""
        from pipeline_transcriber.stages.export import ExportStage

        job = make_job()
        cfg = make_config(tmp_path)
        cfg.export.formats = ["txt"]  # only txt, no srt/vtt
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "artifacts").mkdir(parents=True)

        ctx = StageContext(
            job=job, config=cfg, job_dir=job_dir,
            batch_id="b1", trace_id="t1",
        )
        ctx.asr_result = {"segments": [{"text": "hi", "start": 0, "end": 1}]}

        stage = ExportStage()
        stage.run(ctx)

        final = json.loads((job_dir / "final.json").read_text())
        assert "txt" in final["artifacts"]
        assert "srt" not in final["artifacts"]
        assert "vtt" not in final["artifacts"]


# ---------------------------------------------------------------------------
# Fix 6: batch_report per batch_id
# ---------------------------------------------------------------------------

class TestBatchReportPerBatchId:
    """Batch report saved with batch_id in filename."""

    def test_per_batch_report_created(self, tmp_path: Path):
        cfg = make_config(tmp_path)

        class OkStage(BaseStage):
            @property
            def stage_name(self):
                return StageName.INPUT_VALIDATE
            def run(self, ctx):
                return StageResult(status=StageStatus.SUCCESS)
            def validate(self, ctx, result):
                return ValidationResult(ok=True)

        def _mock_build(config, job=None):
            return [OkStage()]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _mock_build):
            orch = Orchestrator(cfg, batch_id="test123")
            orch.run_batch([make_job()])

        work_dir = Path(cfg.app.work_dir)
        assert (work_dir / "batch_report_test123.json").exists()
        assert (work_dir / "batch_report.json").exists()

        per_batch = json.loads((work_dir / "batch_report_test123.json").read_text())
        assert per_batch["batch_id"] == "test123"
