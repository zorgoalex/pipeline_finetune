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


class DiarizeStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.SPEAKER_DIARIZATION

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        diar_dir = ctx.artifacts_dir / "diarization"
        diar_dir.mkdir(parents=True, exist_ok=True)

        # Mock RTTM lines
        rttm_lines = [
            "SPEAKER audio 1 0.000 5.000 <NA> <NA> SPEAKER_00 <NA> <NA>",
            "SPEAKER audio 1 6.000 5.000 <NA> <NA> SPEAKER_01 <NA> <NA>",
            "SPEAKER audio 1 12.000 3.000 <NA> <NA> SPEAKER_00 <NA> <NA>",
        ]

        rttm_path = diar_dir / "diarization_raw.rttm"
        rttm_path.write_text("\n".join(rttm_lines) + "\n")

        diar_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 6.0, "end": 11.0, "speaker": "SPEAKER_01"},
            {"start": 12.0, "end": 15.0, "speaker": "SPEAKER_00"},
        ]

        segments_path = diar_dir / "diarization_segments.json"
        segments_path.write_text(json.dumps(diar_segments, indent=2))

        report_path = diar_dir / "diarization_report.json"
        report = {
            "backend": ctx.config.diarization.backend,
            "num_speakers": 2,
            "num_segments": len(diar_segments),
        }
        report_path.write_text(json.dumps(report, indent=2))

        ctx.diarization_result = {"segments": diar_segments, "num_speakers": 2}

        artifacts = [str(rttm_path), str(segments_path), str(report_path)]

        log.info(num_speakers=2)
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_speakers": 2, "num_segments": len(diar_segments)},
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
