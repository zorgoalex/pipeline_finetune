"""Job state management for checkpointing/resume."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        self.failed_stages: dict[str, dict[str, Any]] = {}
        self.stage_ledger: list[dict[str, Any]] = []
        self.updated_at: str | None = None
        # Canonical state fields
        self.execution_plan: list[str] = []
        self.config_hash: str = ""
        self.job_snapshot: dict[str, Any] = {}

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

    def mark_stage_failed(self, stage_name: str, attempts: int, error: str) -> None:
        """Record that a stage failed."""
        self.failed_stages[stage_name] = {
            "attempts": attempts,
            "error": error,
        }
        self.current_stage = None
        self._save()

    def mark_job_finished(self, status: str) -> None:
        """Mark the entire job as finished with a terminal status."""
        self.status = status
        self.current_stage = None
        self._save()

    def set_ledger(self, ledger_data: list[dict[str, Any]]) -> None:
        """Store the full stage execution ledger."""
        self.stage_ledger = ledger_data
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write current state to *state.json* atomically via tmp + os.replace."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": self.job_id,
            "completed_stages": self.completed_stages,
            "current_stage": self.current_stage,
            "status": self.status,
            "stage_attempts": self.stage_attempts,
            "failed_stages": self.failed_stages,
            "stage_ledger": self.stage_ledger,
            "execution_plan": self.execution_plan,
            "config_hash": self.config_hash,
            "job_snapshot": self.job_snapshot,
            "updated_at": self.updated_at,
        }
        tmp_path = self.state_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp_path, self.state_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

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
            state.failed_stages = data.get("failed_stages", {})
            state.stage_ledger = data.get("stage_ledger", [])
            state.execution_plan = data.get("execution_plan", [])
            state.config_hash = data.get("config_hash", "")
            state.job_snapshot = data.get("job_snapshot", {})
            state.updated_at = data.get("updated_at")
        return state

    @staticmethod
    def compute_config_hash(config_dict: dict[str, Any]) -> str:
        """SHA-256 hash of serialized config for drift detection."""
        serialized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
