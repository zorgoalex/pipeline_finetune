from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

        if ctx.audio_path is None:
            raise RuntimeError("No audio path available for ASR")

        asr_dir = ctx.artifacts_dir / "asr"
        asr_dir.mkdir(parents=True, exist_ok=True)

        asr_config = ctx.config.asr
        device = self._resolve_device(asr_config.device)

        log.info(
            "asr_loading_model",
            engine=asr_config.engine,
            model=asr_config.model_name,
            device=device,
            compute_type=asr_config.compute_type,
        )

        import whisperx

        model = whisperx.load_model(
            asr_config.model_name,
            device=device,
            compute_type=asr_config.compute_type,
            language=ctx.job.language if ctx.job.language != "auto" else None,
        )

        log.info("asr_transcribing", audio_path=str(ctx.audio_path))
        audio = whisperx.load_audio(str(ctx.audio_path))
        result = model.transcribe(
            audio,
            batch_size=asr_config.batch_size,
            language=ctx.job.language if ctx.job.language != "auto" else None,
        )

        segments = result.get("segments", [])
        detected_language = result.get("language", ctx.job.language)

        asr_result: dict[str, Any] = {
            "segments": segments,
            "language": detected_language,
        }

        # Save raw ASR output
        raw_path = asr_dir / "raw_asr.json"
        raw_path.write_text(json.dumps(asr_result, indent=2, ensure_ascii=False))

        # Save segments as JSONL
        jsonl_path = asr_dir / "asr_segments.jsonl"
        with jsonl_path.open("w") as fh:
            for seg in segments:
                fh.write(json.dumps(seg, ensure_ascii=False) + "\n")

        # Save report
        report_path = asr_dir / "asr_report.json"
        report = {
            "engine": asr_config.engine,
            "model": asr_config.model_name,
            "device": device,
            "compute_type": asr_config.compute_type,
            "language_detected": detected_language,
            "num_segments": len(segments),
            "total_duration": segments[-1]["end"] if segments else 0,
        }
        report_path.write_text(json.dumps(report, indent=2))

        ctx.asr_result = asr_result

        artifacts = [str(raw_path), str(jsonl_path), str(report_path)]
        log.info(
            "asr_complete",
            num_segments=len(segments),
            language=detected_language,
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_segments": len(segments), "language": detected_language},
        )

    def _resolve_device(self, device_setting: str) -> str:
        if device_setting != "auto":
            return device_setting
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

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

    def suggest_fallback(self, attempt: int, ctx: StageContext) -> dict[str, Any]:
        """On retry, try smaller model or lower batch size."""
        asr_cfg = ctx.config.asr
        if attempt == 2 and asr_cfg.batch_size > 4:
            asr_cfg.batch_size = max(4, asr_cfg.batch_size // 2)
            return {"action": "reduce_batch_size", "new_batch_size": asr_cfg.batch_size}
        if attempt >= 3 and asr_cfg.model_name != "tiny":
            model_fallback = {"large-v3": "medium", "medium": "small", "small": "base", "base": "tiny"}
            new_model = model_fallback.get(asr_cfg.model_name, "tiny")
            asr_cfg.model_name = new_model
            return {"action": "downgrade_model", "new_model": new_model}
        return {}
