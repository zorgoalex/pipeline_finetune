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


class AssignSpeakersStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.SPEAKER_ASSIGNMENT

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        fusion_dir = ctx.artifacts_dir / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)

        # Merge aligned result with diarization speaker labels
        aligned_segments = (ctx.aligned_result or {}).get("segments", [])
        diar_segments = (ctx.diarization_result or {}).get("segments", [])

        fused_segments = []
        for seg in aligned_segments:
            speaker = "UNKNOWN"
            for dseg in diar_segments:
                if dseg["start"] <= seg["start"] and seg["end"] <= dseg["end"]:
                    speaker = dseg["speaker"]
                    break
            fused_segments.append({**seg, "speaker": speaker})

        fused_result = {
            "segments": fused_segments,
            "language": (ctx.aligned_result or {}).get("language", "auto"),
            "num_speakers": (ctx.diarization_result or {}).get("num_speakers", 0),
        }

        fused_path = fusion_dir / "fused_result.json"
        fused_path.write_text(json.dumps(fused_result, indent=2))

        report_path = fusion_dir / "fusion_report.json"
        assigned_count = sum(1 for s in fused_segments if s["speaker"] != "UNKNOWN")
        report = {
            "total_segments": len(fused_segments),
            "assigned_segments": assigned_count,
        }
        report_path.write_text(json.dumps(report, indent=2))

        ctx.fused_result = fused_result

        artifacts = [str(fused_path), str(report_path)]

        log.info(assigned=assigned_count)
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"assigned_segments": assigned_count},
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
        return ValidationResult(ok=all_ok, checks=checks, next_stage_allowed=all_ok)
