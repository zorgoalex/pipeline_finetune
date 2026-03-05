"""Job state management for checkpointing/resume."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


STATE_FILENAME = "state.json"


class JobState:
    """Tracks per-job progress so runs can be resumed after failure."""

    def __init__(self, job_id: str, job_dir: Path) -> None:
        self.job_id = job_id
        self.job_dir = job_dir
        self.state_path = job_dir / STATE_FILENAME
        self.completed_stages: list[str] = []
        self.current_stage: str | None = None
        self.status: str = "pending"
        self.stage_attempts: dict[str, int] = {}
        self.updated_at: str | None = None

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def mark_stage_started(self, stage_name: str) -> None:
        """Record that a stage is now running."""
        self.current_stage = stage_name
        self.status = "running"
        self._save()

    def mark_stage_completed(self, stage_name: str, attempts: int) -> None:
        """Record successful completion of a stage."""
        self.completed_stages.append(stage_name)
        self.stage_attempts[stage_name] = attempts
        self.current_stage = None
        self._save()

    def mark_job_finished(self, status: str) -> None:
        """Mark the entire job as finished with a terminal status."""
        self.status = status
        self.current_stage = None
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write current state to *state.json*."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": self.job_id,
            "completed_stages": self.completed_stages,
            "current_stage": self.current_stage,
            "status": self.status,
            "stage_attempts": self.stage_attempts,
            "updated_at": self.updated_at,
        }
        self.state_path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, job_id: str, job_dir: Path) -> JobState:
        """Load an existing state file or return a fresh :class:`JobState`."""
        state = cls(job_id, job_dir)
        state_path = job_dir / STATE_FILENAME
        if state_path.exists():
            data = json.loads(state_path.read_text())
            state.completed_stages = data.get("completed_stages", [])
            state.current_stage = data.get("current_stage")
            state.status = data.get("status", "pending")
            state.stage_attempts = data.get("stage_attempts", {})
            state.updated_at = data.get("updated_at")
        return state
