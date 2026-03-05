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


class AudioPrepareStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.AUDIO_PREPARE

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        audio_dir = ctx.artifacts_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / "audio_16k_mono.wav"
        audio_path.write_bytes(b"")

        probe_path = audio_dir / "audio_probe.json"
        probe = {
            "sample_rate": 16000,
            "channels": 1,
            "duration_sec": 15.0,
            "codec": "pcm_s16le",
        }
        probe_path.write_text(json.dumps(probe, indent=2))

        ctx.audio_path = audio_path

        artifacts = [str(audio_path), str(probe_path)]

        log.info(artifacts=artifacts)
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
