from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StageName(str, Enum):
    INPUT_VALIDATE = "INPUT_VALIDATE"
    DOWNLOAD = "DOWNLOAD"
    AUDIO_PREPARE = "AUDIO_PREPARE"
    VAD_SEGMENTATION = "VAD_SEGMENTATION"
    ASR_TRANSCRIPTION = "ASR_TRANSCRIPTION"
    ALIGNMENT = "ALIGNMENT"
    SPEAKER_DIARIZATION = "SPEAKER_DIARIZATION"
    SPEAKER_ASSIGNMENT = "SPEAKER_ASSIGNMENT"
    EXPORTER = "EXPORTER"
    QA_VALIDATOR = "QA_VALIDATOR"
    FINALIZE_REPORT = "FINALIZE_REPORT"


class StageStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class StageError(BaseModel):
    error_type: str
    message: str
    traceback: str | None = Field(default=None)
    retryable: bool = Field(default=True)


class StageResult(BaseModel):
    status: StageStatus
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: StageError | None = Field(default=None)


class CheckResult(BaseModel):
    name: str
    passed: bool
    details: str = Field(default="")


class ValidationResult(BaseModel):
    ok: bool
    checks: list[CheckResult] = Field(default_factory=list)
    retry_recommended: bool = Field(default=False)
    retry_reason: str | None = Field(default=None)
    next_stage_allowed: bool = Field(default=True)
