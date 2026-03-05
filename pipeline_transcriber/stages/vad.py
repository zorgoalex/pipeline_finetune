from __future__ import annotations

import json
from pathlib import Path

from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import BaseStage, StageContext


class VadStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.VAD_SEGMENTATION

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        vad_dir = ctx.artifacts_dir / "vad"
        vad_dir.mkdir(parents=True, exist_ok=True)

        segments = [
            {"start": 0, "end": 5},
            {"start": 6, "end": 11},
            {"start": 12, "end": 15},
        ]

        segments_path = vad_dir / "vad_segments.json"
        segments_path.write_text(json.dumps(segments, indent=2))

        ctx.vad_segments = segments

        artifacts = [str(segments_path)]

        log.info(num_segments=len(segments))
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_segments": len(segments)},
        )

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        checks: list[CheckResult] = []
        all_ok = True

        for artifact in result.artifacts:
            exists = Path(artifact).exists()
            checks.append(
                CheckResult(
                    name=f"file_exists:{Path(artifact).name}",
                    passed=exists,
                    details=f"{artifact} exists={exists}",
                )
            )
            if not exists:
                all_ok = False

        has_segments = ctx.vad_segments is not None and len(ctx.vad_segments) > 0
        checks.append(
            CheckResult(
                name="segments_non_empty",
                passed=has_segments,
                details=f"VAD produced {len(ctx.vad_segments) if ctx.vad_segments else 0} segments.",
            )
        )
        if not has_segments:
            all_ok = False

        return ValidationResult(ok=all_ok, checks=checks, next_stage_allowed=all_ok)
