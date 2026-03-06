"""Execution contract models: requested → effective → actual."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExecutionPlan(BaseModel):
    """Built at job start: what was requested and what the pipeline decided to do."""

    # Requested (from job)
    requested_output_formats: list[str] = Field(default_factory=list)
    requested_diarization: bool = False
    requested_word_timestamps: bool = True
    requested_speakers: dict[str, int] | None = None  # {min, max} or None
    requested_language: str = "auto"

    # Effective (resolved from job + config)
    effective_output_formats: list[str] = Field(default_factory=list)
    effective_stages: list[str] = Field(default_factory=list)
    effective_speaker_bounds: dict[str, Any] | None = None  # {min, max, source}
    effective_alignment_enabled: bool = False
    effective_diarization_enabled: bool = False


class ExecutionOutcome(BaseModel):
    """Built at finalization: what actually happened.

    ``artifacts_written`` only includes artifacts from successful stages.
    """

    status: str = "pending"
    stages_executed: list[dict[str, Any]] = Field(default_factory=list)
    timings_type: str | None = None  # "word" | "segment"
    num_speakers: int | None = None
    num_segments: int = 0
    artifacts_written: list[str] = Field(default_factory=list)
    failed_stage: str | None = None
    error_message: str | None = None
    error_type: str | None = None
