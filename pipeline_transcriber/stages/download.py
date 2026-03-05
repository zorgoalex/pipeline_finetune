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


class DownloadStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.DOWNLOAD

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        raw_dir = ctx.artifacts_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        media_path = raw_dir / "source_media.wav"
        media_path.write_bytes(b"")

        meta_path = raw_dir / "source_meta.json"
        meta = {
            "source": ctx.job.source,
            "source_type": ctx.job.source_type,
            "title": "Mock Title",
            "duration_sec": 15.0,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        ctx.download_output_path = media_path

        artifacts = [str(media_path), str(meta_path)]

        log.info("download_complete", artifacts=artifacts)
        return StageResult(status=StageStatus.SUCCESS, artifacts=artifacts)

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
