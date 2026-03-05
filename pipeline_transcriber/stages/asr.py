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


class AsrStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.ASR_TRANSCRIPTION

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        asr_dir = ctx.artifacts_dir / "asr"
        asr_dir.mkdir(parents=True, exist_ok=True)

        segments = [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Mock segment 1"},
            {"id": 1, "start": 6.0, "end": 11.0, "text": "Mock segment 2"},
            {"id": 2, "start": 12.0, "end": 15.0, "text": "Mock segment 3"},
        ]

        asr_result = {"segments": segments, "language": ctx.job.language}

        raw_path = asr_dir / "raw_asr.json"
        raw_path.write_text(json.dumps(asr_result, indent=2))

        jsonl_path = asr_dir / "asr_segments.jsonl"
        with jsonl_path.open("w") as fh:
            for seg in segments:
                fh.write(json.dumps(seg) + "\n")

        report_path = asr_dir / "asr_report.json"
        report = {
            "engine": ctx.config.asr.engine,
            "model": ctx.config.asr.model_name,
            "num_segments": len(segments),
        }
        report_path.write_text(json.dumps(report, indent=2))

        ctx.asr_result = asr_result

        artifacts = [str(raw_path), str(jsonl_path), str(report_path)]

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

        has_segments = (
            ctx.asr_result is not None
            and len(ctx.asr_result.get("segments", [])) > 0
        )
        checks.append(
            CheckResult(
                name="asr_segments_non_empty",
                passed=has_segments,
                details="ASR produced non-empty segments."
                if has_segments
                else "ASR produced no segments.",
            )
        )
        if not has_segments:
            all_ok = False

        return ValidationResult(ok=all_ok, checks=checks, next_stage_allowed=all_ok)
