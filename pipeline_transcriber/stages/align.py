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


class AlignStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.ALIGNMENT

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        align_dir = ctx.artifacts_dir / "alignment"
        align_dir.mkdir(parents=True, exist_ok=True)

        # Build aligned result from ASR result, adding mock word-level data
        asr_segments = (ctx.asr_result or {}).get("segments", [])
        aligned_segments = []
        all_words = []
        for seg in asr_segments:
            words = [
                {"word": w, "start": seg["start"], "end": seg["end"]}
                for w in seg.get("text", "").split()
            ]
            aligned_seg = {**seg, "words": words}
            aligned_segments.append(aligned_seg)
            all_words.extend(words)

        aligned_result = {
            "segments": aligned_segments,
            "language": (ctx.asr_result or {}).get("language", "auto"),
        }

        result_path = align_dir / "aligned_result.json"
        result_path.write_text(json.dumps(aligned_result, indent=2))

        words_path = align_dir / "words.jsonl"
        with words_path.open("w") as fh:
            for w in all_words:
                fh.write(json.dumps(w) + "\n")

        report_path = align_dir / "alignment_report.json"
        report = {
            "num_segments": len(aligned_segments),
            "num_words": len(all_words),
        }
        report_path.write_text(json.dumps(report, indent=2))

        ctx.aligned_result = aligned_result

        artifacts = [str(result_path), str(words_path), str(report_path)]

        log.info(num_words=len(all_words))
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_words": len(all_words), "num_segments": len(aligned_segments)},
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
