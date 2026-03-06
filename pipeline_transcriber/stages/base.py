from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import structlog

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.execution import ExecutionPlan
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.models.stage import StageName, StageEntry, StageResult, ValidationResult


@dataclass
class StageContext:
    """Mutable context passed through stages for a single job."""
    job: Job
    config: PipelineConfig
    job_dir: Path
    batch_id: str
    trace_id: str
    stage_outputs: dict[str, StageResult] = field(default_factory=dict)
    stage_ledger: list[StageEntry] = field(default_factory=list)
    execution_plan: ExecutionPlan | None = None
    # Typed inter-stage data
    download_output_path: Path | None = None
    audio_path: Path | None = None
    vad_segments: list[dict[str, Any]] | None = None
    asr_result: dict[str, Any] | None = None
    aligned_result: dict[str, Any] | None = None
    diarization_result: dict[str, Any] | None = None
    fused_result: dict[str, Any] | None = None

    @property
    def artifacts_dir(self) -> Path:
        return self.job_dir / "artifacts"


class BaseStage(ABC):
    @property
    @abstractmethod
    def stage_name(self) -> StageName: ...

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult: ...

    @abstractmethod
    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult: ...

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return True

    def suggest_fallback(self, attempt: int, ctx: StageContext) -> dict[str, Any]:
        return {}

    def cleanup_temp(self, ctx: StageContext) -> None:
        pass

    def _log(self, ctx: StageContext) -> structlog.stdlib.BoundLogger:
        return structlog.get_logger().bind(
            job_id=ctx.job.job_id, stage=self.stage_name.value,
            batch_id=ctx.batch_id, trace_id=ctx.trace_id,
        )
