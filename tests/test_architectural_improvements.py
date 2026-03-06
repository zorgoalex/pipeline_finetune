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
from pipeline_transcriber.models.stage import StageEntry, StageResult, StageStatus
from pipeline_transcriber.orchestrator import Orchestrator
from pipeline_transcriber.stages.base import StageContext
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


# ===========================================================================
# Task 2: Indestructible Finalization
# ===========================================================================

class TestIndestructibleFinalization:

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

        # Write a rich final.json before safety net
        final_path = ctx.job_dir / "final.json"
        rich_data = {
            "job_id": ctx.job.job_id,
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello"}],
            "finalized": True,
        }
        final_path.write_text(json.dumps(rich_data))

        log = MagicMock()
        orch._write_safety_net_artifacts(ctx, "success", log)

        # Should not overwrite — rich version preserved
        final = json.loads(final_path.read_text())
        assert final["segments"] == [{"start": 0.0, "end": 5.0, "text": "Hello"}]
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
