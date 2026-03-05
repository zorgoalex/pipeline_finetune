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
from pipeline_transcriber.utils.ffmpeg import extract_audio, probe_audio


class AudioPrepareStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.AUDIO_PREPARE

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        if ctx.download_output_path is None:
            raise RuntimeError("No download output path available")

        audio_dir = ctx.artifacts_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg_cfg = ctx.config.ffmpeg
        audio_path = audio_dir / "audio_16k_mono.wav"

        extract_audio(
            input_path=ctx.download_output_path,
            output_path=audio_path,
            sample_rate=ffmpeg_cfg.audio_sr,
            channels=ffmpeg_cfg.audio_channels,
            ffmpeg_path=ffmpeg_cfg.ffmpeg_path,
            normalize=ffmpeg_cfg.normalize_audio,
        )

        probe = probe_audio(audio_path, ffprobe_path=ffmpeg_cfg.ffprobe_path)

        probe_path = audio_dir / "audio_probe.json"
        probe_path.write_text(json.dumps(probe, indent=2))

        # Validate duration: output should be within 2s of source if source duration known
        source_meta_path = ctx.artifacts_dir / "raw" / "source_meta.json"
        if source_meta_path.exists():
            source_meta = json.loads(source_meta_path.read_text())
            source_dur = source_meta.get("duration_sec")
            if source_dur is not None:
                diff = abs(probe["duration_sec"] - float(source_dur))
                if diff > 2.0:
                    log.warning(
                        "duration_mismatch",
                        source_duration=source_dur,
                        output_duration=probe["duration_sec"],
                        diff=diff,
                    )

        ctx.audio_path = audio_path

        artifacts = [str(audio_path), str(probe_path)]
        log.info("audio_prepare_complete", artifacts=artifacts, probe=probe)
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={
                "sample_rate": probe["sample_rate"],
                "channels": probe["channels"],
                "duration_sec": probe["duration_sec"],
            },
        )

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        checks: list[CheckResult] = []
        all_ok = True

        for artifact in result.artifacts:
            p = Path(artifact)
            exists = p.exists()
            checks.append(
                CheckResult(
                    name=f"file_exists:{p.name}",
                    passed=exists,
                    details=f"{artifact} exists={exists}",
                )
            )
            if not exists:
                all_ok = False

        # Verify audio properties
        if ctx.audio_path and ctx.audio_path.exists():
            try:
                probe = probe_audio(
                    ctx.audio_path,
                    ffprobe_path=ctx.config.ffmpeg.ffprobe_path,
                )
                sr_ok = probe["sample_rate"] == ctx.config.ffmpeg.audio_sr
                ch_ok = probe["channels"] == ctx.config.ffmpeg.audio_channels
                checks.append(
                    CheckResult(
                        name="sample_rate_correct",
                        passed=sr_ok,
                        details=f"Expected {ctx.config.ffmpeg.audio_sr}, got {probe['sample_rate']}",
                    )
                )
                checks.append(
                    CheckResult(
                        name="channels_correct",
                        passed=ch_ok,
                        details=f"Expected {ctx.config.ffmpeg.audio_channels}, got {probe['channels']}",
                    )
                )
                if not sr_ok or not ch_ok:
                    all_ok = False
            except Exception:
                checks.append(
                    CheckResult(
                        name="probe_validation",
                        passed=False,
                        details="Failed to probe output audio",
                    )
                )
                all_ok = False

        return ValidationResult(ok=all_ok, checks=checks, next_stage_allowed=all_ok)
