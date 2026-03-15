"""Tests for the three architectural improvements:
1. Canonical State Model
2. Indestructible Finalization
3. Execution Contract
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.execution import ExecutionOutcome, ExecutionPlan
from pipeline_transcriber.models.job import ExpectedSpeakers, Job
from pipeline_transcriber.models.stage import (
    StageEntry,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.orchestrator import Orchestrator
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.stages.finalize import FinalizeReportStage
from pipeline_transcriber.utils.state import JobState


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
        enable_word_timestamps=False,
    )
    defaults.update(kwargs)
    return Job(**defaults)


def make_ctx(tmp_path: Path, job: Job | None = None, config: PipelineConfig | None = None) -> StageContext:
    job = job or make_job()
    config = config or make_config(tmp_path)
    job_dir = tmp_path / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        job=job,
        config=config,
        job_dir=job_dir,
        batch_id="test-batch",
        trace_id="test-trace",
    )


# ===========================================================================
# Task 1: Canonical State Model
# ===========================================================================

class TestCanonicalStateModel:

    def test_state_atomic_write(self, tmp_path: Path):
        """JobState._save() creates file atomically — no .tmp left behind."""
        job_dir = tmp_path / "job01"
        job_dir.mkdir()
        state = JobState("job01", job_dir)
        state.mark_stage_started("DOWNLOAD")

        # state.json should exist, .tmp should not
        assert (job_dir / "state.json").exists()
        assert not (job_dir / "state.json.tmp").exists()

    def test_state_new_fields_persist(self, tmp_path: Path):
        """execution_plan, config_hash, job_snapshot survive save+load cycle."""
        job_dir = tmp_path / "job02"
        job_dir.mkdir()
        state = JobState("job02", job_dir)
        state.execution_plan = ["INPUT_VALIDATE", "DOWNLOAD", "AUDIO_PREPARE"]
        state.config_hash = "abc123def456"
        state.job_snapshot = {"job_id": "job02", "source": "/tmp/x.wav"}
        state._save()

        loaded = JobState.load("job02", job_dir)
        assert loaded.execution_plan == ["INPUT_VALIDATE", "DOWNLOAD", "AUDIO_PREPARE"]
        assert loaded.config_hash == "abc123def456"
        assert loaded.job_snapshot == {"job_id": "job02", "source": "/tmp/x.wav"}

    def test_config_hash_deterministic(self):
        """Same config dict produces same hash."""
        cfg = {"model": "large-v3", "language": "kk", "batch_size": 16}
        h1 = JobState.compute_config_hash(cfg)
        h2 = JobState.compute_config_hash(cfg)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_config_hash_detects_change(self):
        """Different config dict produces different hash."""
        cfg_a = {"model": "large-v3", "language": "kk"}
        cfg_b = {"model": "large-v3", "language": "en"}
        assert JobState.compute_config_hash(cfg_a) != JobState.compute_config_hash(cfg_b)

    def test_resume_ledger_restored_as_stage_entry(self, tmp_path: Path):
        """On resume, state.stage_ledger dicts become StageEntry objects in ctx."""
        job_dir = tmp_path / "job03"
        job_dir.mkdir()
        state = JobState("job03", job_dir)
        state.stage_ledger = [
            {"stage_name": "INPUT_VALIDATE", "status": "success", "attempts": 1, "duration_ms": 50},
            {"stage_name": "DOWNLOAD", "status": "success", "attempts": 2, "duration_ms": 3000},
        ]
        state._save()

        # Simulate resume: load state and restore ledger into ctx
        loaded = JobState.load("job03", job_dir)
        assert len(loaded.stage_ledger) == 2

        # The orchestrator converts these dicts to StageEntry on resume
        entries = [StageEntry.model_validate(e) for e in loaded.stage_ledger]
        assert all(isinstance(e, StageEntry) for e in entries)
        assert entries[0].stage_name == "INPUT_VALIDATE"
        assert entries[1].attempts == 2

    def test_resume_execution_plan_intersection(self, tmp_path: Path):
        """Completed stages not in new plan get pruned."""
        job_dir = tmp_path / "job04"
        job_dir.mkdir()

        # Simulate old state with completed stages
        state = JobState("job04", job_dir)
        state.completed_stages = ["INPUT_VALIDATE", "DOWNLOAD", "OLD_STAGE"]
        state._save()

        loaded = JobState.load("job04", job_dir)

        # Simulate the orchestrator's pruning logic
        new_plan = ["INPUT_VALIDATE", "DOWNLOAD", "AUDIO_PREPARE", "ASR_TRANSCRIPTION"]
        new_plan_set = set(new_plan)
        pruned = [s for s in loaded.completed_stages if s not in new_plan_set]
        loaded.completed_stages = [s for s in loaded.completed_stages if s in new_plan_set]

        assert pruned == ["OLD_STAGE"]
        assert loaded.completed_stages == ["INPUT_VALIDATE", "DOWNLOAD"]

    def test_job_hash_persists(self, tmp_path: Path):
        """job_hash survives save+load cycle."""
        job_dir = tmp_path / "job05"
        job_dir.mkdir()
        state = JobState("job05", job_dir)

        job_dict = {"job_id": "job05", "source": "/tmp/test.wav", "language": "kk"}
        state.job_hash = JobState.compute_config_hash(job_dict)
        state._save()

        loaded = JobState.load("job05", job_dir)
        assert loaded.job_hash == state.job_hash
        assert len(loaded.job_hash) == 64  # SHA-256 hex digest

    def test_resume_job_drift_critical_invalidation(self, tmp_path: Path):
        """When source changes between runs, completed_stages cleared."""
        job_dir = tmp_path / "job06"
        job_dir.mkdir()

        # Simulate first run: save state with completed stages and a job snapshot
        old_job_dict = {"job_id": "job06", "source": "/tmp/old.wav", "source_type": "local_file", "language": "kk"}
        state = JobState("job06", job_dir)
        state.completed_stages = ["INPUT_VALIDATE", "DOWNLOAD", "AUDIO_PREPARE"]
        state.job_hash = JobState.compute_config_hash(old_job_dict)
        state.job_snapshot = old_job_dict
        state._save()

        # Simulate resume with changed critical field (source)
        loaded = JobState.load("job06", job_dir)
        new_job_dict = {"job_id": "job06", "source": "/tmp/new.wav", "source_type": "local_file", "language": "kk"}
        current_job_hash = JobState.compute_config_hash(new_job_dict)

        # Replicate the orchestrator's drift detection logic
        assert loaded.job_hash != current_job_hash

        critical_fields = ("source", "source_type", "language",
                           "enable_diarization", "enable_word_timestamps")
        critical_changed = [
            f for f in critical_fields
            if loaded.job_snapshot.get(f) != new_job_dict.get(f)
        ]
        assert critical_changed == ["source"]

        # Invalidate completed stages (as orchestrator does)
        loaded.completed_stages = []
        assert loaded.completed_stages == []


# ===========================================================================
# Task 2: Indestructible Finalization
# ===========================================================================

class TestIndestructibleFinalization:

    @staticmethod
    def _make_ok_stage(name: StageName, call_counts: dict[str, int] | None = None) -> BaseStage:
        class _Stage(BaseStage):
            @property
            def stage_name(self) -> StageName:
                return name

            def run(self, ctx):
                if call_counts is not None:
                    call_counts[name.value] = call_counts.get(name.value, 0) + 1
                return StageResult(status=StageStatus.SUCCESS)

            def validate(self, ctx, result):
                return ValidationResult(ok=True, checks=[])

        return _Stage()

    def test_safety_net_writes_minimal_report(self, tmp_path: Path):
        """_write_safety_net_artifacts creates report.json with finalized=False."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success"),
        ]

        log = MagicMock()
        orch._write_safety_net_artifacts(ctx, "success", log)

        report_path = ctx.job_dir / "report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["finalized"] is False
        assert report["status"] == "success"
        assert report["job_id"] == ctx.job.job_id

    def test_safety_net_writes_minimal_final(self, tmp_path: Path):
        """Creates final.json with finalized=False when no rich version exists."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = []

        log = MagicMock()
        orch._write_safety_net_artifacts(ctx, "failed", log)

        final_path = ctx.job_dir / "final.json"
        assert final_path.exists()
        final = json.loads(final_path.read_text())
        assert final["finalized"] is False
        assert final["status"] == "failed"
        assert final["job_id"] == ctx.job.job_id

    def test_safety_net_skips_rich_final(self, tmp_path: Path):
        """Does NOT overwrite final.json that has non-empty segments."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = []

        # Write a rich final.json with ExportStage keys (even if segments=[])
        final_path = ctx.job_dir / "final.json"
        rich_data = {
            "job_id": ctx.job.job_id,
            "segments": [],
            "pipeline": {"version": "0.1.0"},
            "audio": {"duration_sec": 10.0},
            "artifacts": {"srt": "transcript.srt"},
            "finalized": True,
        }
        final_path.write_text(json.dumps(rich_data))

        log = MagicMock()
        orch._write_safety_net_artifacts(ctx, "success", log)

        # Should not overwrite — rich version preserved (even with empty segments)
        final = json.loads(final_path.read_text())
        assert final["pipeline"] == {"version": "0.1.0"}
        assert final["finalized"] is True

    def test_safety_net_per_file_isolation(self, tmp_path: Path):
        """If report.json write fails (bad path), final.json still gets written."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = []

        log = MagicMock()

        # Make report.json write fail by patching _atomic_write_json to fail on first call only
        original_write = Orchestrator._atomic_write_json
        call_count = [0]

        def failing_write(path, data):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("disk full")
            original_write(path, data)

        with patch.object(Orchestrator, '_atomic_write_json', staticmethod(failing_write)):
            orch._write_safety_net_artifacts(ctx, "failed", log)

        # report.json should NOT exist (write failed)
        assert not (ctx.job_dir / "report.json").exists()

        # final.json SHOULD exist (independent try/except)
        final_path = ctx.job_dir / "final.json"
        assert final_path.exists()
        final = json.loads(final_path.read_text())
        assert final["finalized"] is False

    def test_finalizer_enriches_report(self, tmp_path: Path):
        """FinalizeReportStage sets finalized=True and adds full data."""
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success", duration_ms=10),
            StageEntry(stage_name="DOWNLOAD", status="success", attempts=2, duration_ms=5000),
        ]

        # Write safety-net report first (as orchestrator does)
        report_path = ctx.job_dir / "report.json"
        report_path.write_text(json.dumps({"finalized": False, "status": "success"}))

        finalizer = FinalizeReportStage()
        finalizer.run(ctx, job_status="success")

        report = json.loads(report_path.read_text())
        assert report["finalized"] is True
        assert report["status"] == "success"
        assert report["total_stages"] == 2
        assert len(report["stages"]) == 2
        assert report["stages"][0]["stage"] == "INPUT_VALIDATE"

    def test_finalizer_enriches_final(self, tmp_path: Path):
        """FinalizeReportStage sets finalized=True, stage_ledger, status in final.json."""
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success"),
            StageEntry(stage_name="DOWNLOAD", status="failed", error="timeout"),
        ]

        # Write safety-net final first
        final_path = ctx.job_dir / "final.json"
        final_path.write_text(json.dumps({
            "job_id": ctx.job.job_id,
            "finalized": False,
            "status": "failed",
        }))

        finalizer = FinalizeReportStage()
        finalizer.run(ctx, job_status="failed")

        final = json.loads(final_path.read_text())
        assert final["finalized"] is True
        assert final["status"] == "failed"
        assert "stage_ledger" in final
        assert len(final["stage_ledger"]) == 2
        assert "execution" in final

    def test_finalizer_repair_of_corrupt_report_leaves_repair_warning(self, tmp_path: Path):
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [StageEntry(stage_name="INPUT_VALIDATE", status="success", duration_ms=10)]

        report_path = ctx.job_dir / "report.json"
        report_path.write_text("{corrupt")

        finalizer = FinalizeReportStage()
        finalizer.run(ctx, job_status="success")

        report = json.loads(report_path.read_text())
        assert report["finalized"] is True
        assert "repair_warnings" in report
        assert any(item["artifact"] == "report.json" for item in report["repair_warnings"])

    def test_finalizer_repair_of_corrupt_final_leaves_repair_warning(self, tmp_path: Path):
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [StageEntry(stage_name="INPUT_VALIDATE", status="success", duration_ms=10)]

        final_path = ctx.job_dir / "final.json"
        final_path.write_text("{corrupt")

        finalizer = FinalizeReportStage()
        finalizer.run(ctx, job_status="success")

        final_data = json.loads(final_path.read_text())
        assert final_data["finalized"] is True
        assert "repair_warnings" in final_data
        assert any(item["artifact"] == "final.json" for item in final_data["repair_warnings"])

    def test_finalization_status_success(self, tmp_path: Path):
        """After successful finalize, state.finalization_status == 'success'."""
        job_dir = tmp_path / "fin_ok"
        job_dir.mkdir()
        state = JobState("fin_ok", job_dir)

        ctx = make_ctx(tmp_path, job=make_job(job_id="fin_ok"))
        # Overwrite job_dir to match state
        ctx.job_dir = job_dir
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success", duration_ms=5),
        ]

        # Write safety-net artifacts first
        report_path = job_dir / "report.json"
        report_path.write_text(json.dumps({"finalized": False, "status": "success"}))
        final_path = job_dir / "final.json"
        final_path.write_text(json.dumps({"job_id": "fin_ok", "finalized": False}))

        # Run finalizer and track status as the orchestrator does
        try:
            finalizer = FinalizeReportStage()
            finalizer.run(ctx, job_status="success")
            state.finalization_status = "success"
        except Exception:
            state.finalization_status = "failed"
        state._save()

        loaded = JobState.load("fin_ok", job_dir)
        assert loaded.finalization_status == "success"

    def test_finalization_status_failed(self, tmp_path: Path):
        """When finalizer raises, state.finalization_status == 'failed'."""
        job_dir = tmp_path / "fin_fail"
        job_dir.mkdir()
        state = JobState("fin_fail", job_dir)

        ctx = make_ctx(tmp_path, job=make_job(job_id="fin_fail"))
        ctx.job_dir = job_dir
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success"),
        ]

        # Simulate finalizer raising an exception
        try:
            with patch.object(FinalizeReportStage, 'run', side_effect=RuntimeError("disk exploded")):
                finalizer = FinalizeReportStage()
                finalizer.run(ctx, job_status="success")
            state.finalization_status = "success"
        except Exception:
            state.finalization_status = "failed"
        state._save()

        loaded = JobState.load("fin_fail", job_dir)
        assert loaded.finalization_status == "failed"

    def test_orchestrator_injects_finalizer_for_legacy_sequence(self, tmp_path: Path):
        """Legacy mocked sequences without finalizer still get guaranteed finalization."""
        cfg = make_config(tmp_path)
        job = make_job()

        class OkStage:
            stage_name = StageName.INPUT_VALIDATE

            def run(self, ctx):
                return StageResult(status=StageStatus.SUCCESS)

            def validate(self, ctx, result):
                from pipeline_transcriber.models.stage import ValidationResult
                return ValidationResult(ok=True, checks=[])

            def can_retry(self, error, ctx):
                return True

            def suggest_fallback(self, attempt_no, ctx):
                return {}

        def _legacy_build(config, job=None):
            return [OkStage()]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _legacy_build):
            orch = Orchestrator(cfg)
            report = orch.run_batch([job])

        assert report["success"] == 1
        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state = json.loads(state_path.read_text())
        assert "FINALIZE_REPORT" in state["execution_plan"]
        assert "FINALIZE_REPORT" in state["completed_stages"]
        assert any(e["stage_name"] == "FINALIZE_REPORT" for e in state["stage_ledger"])

    def test_finalizer_failure_visible_without_overriding_main_status(self, tmp_path: Path):
        """Main job status stays honest while finalizer failure is explicit in state/ledger."""
        cfg = make_config(tmp_path)
        job = make_job()

        class OkStage:
            stage_name = StageName.INPUT_VALIDATE

            def run(self, ctx):
                return StageResult(status=StageStatus.SUCCESS)

            def validate(self, ctx, result):
                from pipeline_transcriber.models.stage import ValidationResult
                return ValidationResult(ok=True, checks=[])

            def can_retry(self, error, ctx):
                return True

            def suggest_fallback(self, attempt_no, ctx):
                return {}

        def _legacy_build(config, job=None):
            return [OkStage()]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _legacy_build):
            with patch.object(FinalizeReportStage, "run", side_effect=RuntimeError("disk exploded")):
                orch = Orchestrator(cfg)
                report = orch.run_batch([job])

        assert report["success"] == 1
        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert state["finalization_status"] == "failed"
        final_entry = next(
            entry for entry in state["stage_ledger"]
            if entry["stage_name"] == "FINALIZE_REPORT"
        )
        assert final_entry["status"] == "failed"

    def test_resume_reruns_only_finalization_after_failed_finalizer(self, tmp_path: Path):
        """Resume should skip main stages and rerun only finalization after prior finalizer failure."""
        cfg = make_config(tmp_path)
        job = make_job()
        call_counts: dict[str, int] = {}
        original_run = FinalizeReportStage.run

        def _legacy_build(config, job=None):
            return [self._make_ok_stage(StageName.INPUT_VALIDATE, call_counts)]

        def flaky_finalizer(self, ctx, job_status="success"):
            call_counts["FINALIZE_REPORT"] = call_counts.get("FINALIZE_REPORT", 0) + 1
            if call_counts["FINALIZE_REPORT"] == 1:
                raise RuntimeError("disk exploded")
            return original_run(self, ctx, job_status=job_status)

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _legacy_build):
            with patch.object(FinalizeReportStage, "run", new=flaky_finalizer):
                orch1 = Orchestrator(cfg, batch_id="b1")
                report1 = orch1.run_batch([job])
                assert report1["success"] == 1

                orch2 = Orchestrator(cfg, batch_id="b2")
                report2 = orch2.run_batch([job], resume=True)

        assert report2["success"] == 1
        assert call_counts["INPUT_VALIDATE"] == 1
        assert call_counts["FINALIZE_REPORT"] == 2

        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert state["finalization_status"] == "success"
        assert "FINALIZE_REPORT" in state["completed_stages"]

    def test_resume_invalidates_only_finalization_when_final_artifacts_corrupt(self, tmp_path: Path):
        """Corrupt final/report artifacts should rerun finalization only."""
        cfg = make_config(tmp_path)
        job = make_job()
        call_counts: dict[str, int] = {}
        original_run = FinalizeReportStage.run

        def _legacy_build(config, job=None):
            return [self._make_ok_stage(StageName.INPUT_VALIDATE, call_counts)]

        def counting_finalizer(self, ctx, job_status="success"):
            call_counts["FINALIZE_REPORT"] = call_counts.get("FINALIZE_REPORT", 0) + 1
            return original_run(self, ctx, job_status=job_status)

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _legacy_build):
            with patch.object(FinalizeReportStage, "run", new=counting_finalizer):
                orch1 = Orchestrator(cfg, batch_id="b1")
                report1 = orch1.run_batch([job])
                assert report1["success"] == 1

                job_dir = Path(cfg.app.work_dir) / job.job_id
                (job_dir / "final.json").write_text("{corrupt")
                (job_dir / "report.json").write_text("{corrupt")

                orch2 = Orchestrator(cfg, batch_id="b2")
                report2 = orch2.run_batch([job], resume=True)

        assert report2["success"] == 1
        assert call_counts["INPUT_VALIDATE"] == 1
        assert call_counts["FINALIZE_REPORT"] == 2

        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert state["finalization_status"] == "success"

    def test_safety_net_failure_does_not_block_finalizer(self, tmp_path: Path):
        """Unexpected safety-net failure should not prevent finalization stages from running."""
        cfg = make_config(tmp_path)
        job = make_job()
        original_run = FinalizeReportStage.run
        call_counts: dict[str, int] = {}

        def _legacy_build(config, job=None):
            return [self._make_ok_stage(StageName.INPUT_VALIDATE, call_counts)]

        def counting_finalizer(self, ctx, job_status="success"):
            call_counts["FINALIZE_REPORT"] = call_counts.get("FINALIZE_REPORT", 0) + 1
            return original_run(self, ctx, job_status=job_status)

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _legacy_build):
            with patch.object(Orchestrator, "_write_safety_net_artifacts", side_effect=RuntimeError("safety-net crashed")):
                with patch.object(FinalizeReportStage, "run", new=counting_finalizer):
                    orch = Orchestrator(cfg)
                    report = orch.run_batch([job])

        assert report["success"] == 1
        assert call_counts["INPUT_VALIDATE"] == 1
        assert call_counts["FINALIZE_REPORT"] == 1

        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert state["finalization_status"] == "success"

    def test_partial_status_propagates_into_final_contract(self, tmp_path: Path):
        """Successful finalization should preserve a partial main-pipeline outcome in final artifacts."""
        cfg = make_config(tmp_path)
        cfg.vad.enabled = True
        job = make_job()

        class FailVadStage(BaseStage):
            @property
            def stage_name(self) -> StageName:
                return StageName.VAD_SEGMENTATION

            def run(self, ctx):
                raise RuntimeError("vad exploded")

            def validate(self, ctx, result):
                return ValidationResult(ok=True, checks=[])

            def can_retry(self, error, ctx):
                return False

        def _build(config, job=None):
            return [
                self._make_ok_stage(StageName.INPUT_VALIDATE),
                FailVadStage(),
            ]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _build):
            orch = Orchestrator(cfg)
            report = orch.run_batch([job])

        assert report["partial"] == 1
        job_dir = Path(cfg.app.work_dir) / job.job_id
        final_data = json.loads((job_dir / "final.json").read_text())
        report_data = json.loads((job_dir / "report.json").read_text())
        assert final_data["status"] == "partial"
        assert report_data["status"] == "partial"

    def test_finalizer_failure_writes_failure_feedback(self, tmp_path: Path):
        """Finalizer failure should still create stage feedback with fail status."""
        cfg = make_config(tmp_path)
        job = make_job()

        def _legacy_build(config, job=None):
            return [self._make_ok_stage(StageName.INPUT_VALIDATE)]

        with patch("pipeline_transcriber.orchestrator.build_stage_sequence", _legacy_build):
            with patch.object(FinalizeReportStage, "run", side_effect=RuntimeError("disk exploded")):
                orch = Orchestrator(cfg)
                report = orch.run_batch([job])

        assert report["success"] == 1
        feedback_path = Path(cfg.app.work_dir) / job.job_id / "stage_feedback" / "FINALIZE_REPORT.json"
        feedback = json.loads(feedback_path.read_text())
        assert feedback["status"] == "fail"
        assert feedback["retry_needed"] is True


# ===========================================================================
# Task 3: Execution Contract
# ===========================================================================

class TestExecutionContract:

    def test_build_execution_plan_basic(self, tmp_path: Path):
        """Check all requested/effective fields are populated."""
        cfg = make_config(tmp_path)
        job = make_job()
        stages = ["INPUT_VALIDATE", "DOWNLOAD", "AUDIO_PREPARE", "ASR_TRANSCRIPTION"]

        plan = Orchestrator._build_execution_plan(job, cfg, stages)

        assert isinstance(plan, ExecutionPlan)
        assert plan.requested_output_formats == ["json", "txt"]
        assert plan.requested_diarization is False
        assert plan.requested_word_timestamps is False
        assert plan.effective_stages == stages
        assert plan.effective_output_formats == ["json", "txt"]
        assert plan.effective_diarization_enabled is False

    def test_build_execution_plan_job_speaker_override(self, tmp_path: Path):
        """speaker_bounds.source='job' when job has expected_speakers."""
        cfg = make_config(tmp_path)
        cfg.diarization.enabled = True
        job = make_job(
            enable_diarization=True,
            expected_speakers=ExpectedSpeakers(min=2, max=4),
        )
        stages = ["INPUT_VALIDATE", "DOWNLOAD"]

        plan = Orchestrator._build_execution_plan(job, cfg, stages)

        assert plan.effective_speaker_bounds is not None
        assert plan.effective_speaker_bounds["source"] == "job"
        assert plan.effective_speaker_bounds["min"] == 2
        assert plan.effective_speaker_bounds["max"] == 4

    def test_build_execution_plan_config_speaker_fallback(self, tmp_path: Path):
        """speaker_bounds.source='config' when job has no expected_speakers."""
        cfg = make_config(tmp_path)
        cfg.diarization.enabled = True
        cfg.diarization.min_speakers = 1
        cfg.diarization.max_speakers = 10
        job = make_job(enable_diarization=True)
        stages = ["INPUT_VALIDATE"]

        plan = Orchestrator._build_execution_plan(job, cfg, stages)

        assert plan.effective_speaker_bounds is not None
        assert plan.effective_speaker_bounds["source"] == "config"
        assert plan.effective_speaker_bounds["min"] == 1
        assert plan.effective_speaker_bounds["max"] == 10

    def test_execution_outcome_from_ledger(self, tmp_path: Path):
        """Builds stages_executed, artifacts_written from ledger."""
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [
            StageEntry(
                stage_name="INPUT_VALIDATE", status="success",
                attempts=1, duration_ms=10,
            ),
            StageEntry(
                stage_name="DOWNLOAD", status="success",
                attempts=1, duration_ms=3000,
                artifacts=["/tmp/raw/source.wav"],
            ),
        ]

        outcome = FinalizeReportStage._build_execution_outcome(ctx, "success")

        assert isinstance(outcome, ExecutionOutcome)
        assert len(outcome.stages_executed) == 2
        assert outcome.stages_executed[0]["name"] == "INPUT_VALIDATE"
        assert outcome.stages_executed[1]["name"] == "DOWNLOAD"
        assert "/tmp/raw/source.wav" in outcome.artifacts_written
        assert outcome.status == "success"
        assert outcome.failed_stage is None

    def test_execution_outcome_captures_failure(self, tmp_path: Path):
        """failed_stage and error_message populated."""
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success"),
            StageEntry(
                stage_name="DOWNLOAD", status="failed",
                error="Connection refused",
            ),
        ]

        outcome = FinalizeReportStage._build_execution_outcome(ctx, "failed")

        assert outcome.failed_stage == "DOWNLOAD"
        assert outcome.error_message == "Connection refused"
        assert outcome.status == "failed"

    def test_execution_contract_in_final_json(self, tmp_path: Path):
        """After finalize, final.json has 'execution' key with plan+outcome."""
        ctx = make_ctx(tmp_path)
        cfg = make_config(tmp_path)
        job = ctx.job

        # Set an execution plan on ctx
        ctx.execution_plan = Orchestrator._build_execution_plan(
            job, cfg, ["INPUT_VALIDATE", "DOWNLOAD"],
        )
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success", duration_ms=10),
            StageEntry(stage_name="DOWNLOAD", status="success", duration_ms=2000),
        ]

        # Write minimal final.json first (as safety net would)
        final_path = ctx.job_dir / "final.json"
        final_path.write_text(json.dumps({"job_id": job.job_id, "finalized": False}))

        finalizer = FinalizeReportStage()
        finalizer.run(ctx, job_status="success")

        final = json.loads(final_path.read_text())
        assert "execution" in final
        assert "plan" in final["execution"]
        assert "outcome" in final["execution"]
        assert final["execution"]["outcome"]["status"] == "success"
        assert len(final["execution"]["plan"]["effective_stages"]) == 2

    def test_error_type_captured_in_ledger(self, tmp_path: Path):
        """StageEntry with error_type populated when stage fails."""
        entry = StageEntry(
            stage_name="DOWNLOAD",
            status="failed",
            error="Connection refused",
            error_type="ConnectionError",
        )
        assert entry.error_type == "ConnectionError"
        assert entry.error == "Connection refused"

        # Verify it round-trips through model_dump / model_validate
        dumped = entry.model_dump()
        assert dumped["error_type"] == "ConnectionError"
        restored = StageEntry.model_validate(dumped)
        assert restored.error_type == "ConnectionError"

    def test_error_type_in_execution_outcome(self, tmp_path: Path):
        """_build_execution_outcome reads error_type from ledger."""
        ctx = make_ctx(tmp_path)
        ctx.stage_ledger = [
            StageEntry(stage_name="INPUT_VALIDATE", status="success", duration_ms=10),
            StageEntry(
                stage_name="DOWNLOAD",
                status="failed",
                error="file not found",
                error_type="FileNotFoundError",
            ),
        ]

        outcome = FinalizeReportStage._build_execution_outcome(ctx, "failed")

        assert outcome.failed_stage == "DOWNLOAD"
        assert outcome.error_message == "file not found"
        assert outcome.error_type == "FileNotFoundError"
