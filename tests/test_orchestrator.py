from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.orchestrator import Orchestrator


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
    return cfg


def make_job(job_id: str = "test-job-01") -> Job:
    return Job(
        job_id=job_id,
        source_type="local_file",
        source="/tmp/test.wav",
        output_formats=["json", "txt"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    """Unit tests for the Orchestrator batch runner."""

    def test_batch_success(self, tmp_path: Path) -> None:
        """Run batch with 1 job. Expect success==1 and batch_report.json on disk."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        jobs = [make_job()]

        report = orch.run_batch(jobs)

        assert report["success"] == 1
        assert report["failed"] == 0

        report_path = Path(cfg.app.work_dir) / "batch_report.json"
        assert report_path.exists()

    def test_batch_two_jobs(self, tmp_path: Path) -> None:
        """Run batch with 2 jobs. Both succeed."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        jobs = [make_job("job-a"), make_job("job-b")]

        report = orch.run_batch(jobs)

        assert report["total"] == 2
        assert report["success"] == 2

    def test_state_json_created(self, tmp_path: Path) -> None:
        """After batch run, state.json exists and records success with completed stages."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        job = make_job()

        orch.run_batch([job])

        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        assert state_path.exists()

        state = json.loads(state_path.read_text())
        assert state["status"] == "success"
        assert len(state["completed_stages"]) > 0

    def test_stage_feedback_created(self, tmp_path: Path) -> None:
        """After batch run, stage_feedback/ directory exists with .json files."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)
        job = make_job()

        orch.run_batch([job])

        feedback_dir = Path(cfg.app.work_dir) / job.job_id / "stage_feedback"
        assert feedback_dir.exists()
        assert feedback_dir.is_dir()

        feedback_files = list(feedback_dir.glob("*.json"))
        assert len(feedback_files) > 0

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        """Run batch once, then resume. Second run still succeeds (stages skipped)."""
        cfg = make_config(tmp_path)
        job = make_job()

        # First run
        orch1 = Orchestrator(cfg, batch_id="batch-1")
        report1 = orch1.run_batch([job])
        assert report1["success"] == 1

        # Load state after first run
        state_path = Path(cfg.app.work_dir) / job.job_id / "state.json"
        state_before = json.loads(state_path.read_text())

        # Second run with resume=True
        orch2 = Orchestrator(cfg, batch_id="batch-2")
        report2 = orch2.run_batch([job], resume=True)
        assert report2["success"] == 1

        # State should still reflect success
        state_after = json.loads(state_path.read_text())
        assert state_after["status"] == "success"
        assert state_after["completed_stages"] == state_before["completed_stages"]

    def test_batch_report_structure(self, tmp_path: Path) -> None:
        """Batch report contains all required top-level keys."""
        cfg = make_config(tmp_path)
        orch = Orchestrator(cfg)

        report = orch.run_batch([make_job()])

        required_keys = {"batch_id", "total", "success", "failed", "partial", "timestamp", "jobs"}
        assert required_keys.issubset(report.keys())
