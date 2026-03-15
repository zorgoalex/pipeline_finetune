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

        import time
        import whisperx

        t0_load = time.monotonic()
        model = whisperx.load_model(
            asr_config.model_name,
            device=device,
            compute_type=asr_config.compute_type,
            language=ctx.job.language if ctx.job.language != "auto" else None,
        )
        model_load_ms = round((time.monotonic() - t0_load) * 1000)

        # Support 128 mel bins for large-v3-turbo models (config override or auto-detect)
        if asr_config.n_mels is not None:
            if hasattr(model, "feat_kwargs"):
                model.feat_kwargs["feature_size"] = asr_config.n_mels
        detected_mels = getattr(model, "feat_kwargs", {}).get("feature_size", 80)
        log.info("asr_model_loaded", model_load_time_ms=model_load_ms, n_mels=detected_mels)

        t0_infer = time.monotonic()
        if asr_config.mode == "vad_clips":
            asr_result = self._transcribe_vad_clips(ctx, model, whisperx)
        else:
            asr_result = self._transcribe_full_audio(ctx, model, whisperx)
        inference_ms = round((time.monotonic() - t0_infer) * 1000)

        segments = asr_result.get("segments", [])
        detected_language = asr_result.get("language", ctx.job.language)

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
        audio_duration = segments[-1]["end"] if segments else 0
        inference_sec = inference_ms / 1000.0
        rtf = round(inference_sec / audio_duration, 4) if audio_duration > 0 else None
        report = {
            "engine": asr_config.engine,
            "model": asr_config.model_name,
            "device": device,
            "compute_type": asr_config.compute_type,
            "batch_size": asr_config.batch_size,
            "mode": asr_config.mode,
            "language_detected": detected_language,
            "num_segments": len(segments),
            "total_duration": audio_duration,
            "model_load_time_ms": model_load_ms,
            "inference_time_ms": inference_ms,
            "rtf": rtf,
        }
        if asr_config.mode == "vad_clips":
            report["clips_expected"] = asr_result.get("clips_expected", 0)
            report["clips_processed"] = asr_result.get("clips_processed", 0)
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
            metrics={
                "num_segments": len(segments),
                "language": detected_language,
                "mode": asr_config.mode,
                "clips_expected": asr_result.get("clips_expected", 0),
                "clips_processed": asr_result.get("clips_processed", 0),
            },
        )

    def _transcribe_full_audio(self, ctx: StageContext, model: Any, whisperx: Any) -> dict[str, Any]:
        log = self._log(ctx)
        log.info("asr_transcribing", mode="full_audio", audio_path=str(ctx.audio_path))
        audio = whisperx.load_audio(str(ctx.audio_path))
        result = model.transcribe(
            audio,
            batch_size=ctx.config.asr.batch_size,
            language=ctx.job.language if ctx.job.language != "auto" else None,
        )

        segments = self._normalize_segment_ids(result.get("segments", []))
        return {
            "segments": segments,
            "language": result.get("language", ctx.job.language),
            "mode": "full_audio",
        }

    def _transcribe_vad_clips(self, ctx: StageContext, model: Any, whisperx: Any) -> dict[str, Any]:
        log = self._log(ctx)
        clip_manifest = self._resolve_clip_manifest(ctx)

        merged_segments: list[dict[str, Any]] = []
        manifest_with_stats: list[dict[str, Any]] = []
        detected_language = ctx.job.language

        for clip in clip_manifest:
            clip_path = self._resolve_clip_path(ctx, clip)
            log.info("asr_transcribing_clip", clip_id=clip["clip_id"], clip_path=str(clip_path))
            audio = whisperx.load_audio(str(clip_path))
            result = model.transcribe(
                audio,
                batch_size=ctx.config.asr.batch_size,
                language=ctx.job.language if ctx.job.language != "auto" else None,
            )
            clip_segments = result.get("segments", [])
            if result.get("language"):
                detected_language = result["language"]

            rebased_segments = []
            for seg in clip_segments:
                rebased = dict(seg)
                rebased["start"] = round(float(rebased.get("start", 0.0)) + float(clip["start"]), 3)
                rebased["end"] = round(float(rebased.get("end", 0.0)) + float(clip["start"]), 3)
                rebased["source_clip_id"] = clip["clip_id"]
                rebased_segments.append(rebased)

            merged_segments.extend(rebased_segments)
            manifest_with_stats.append({
                "clip_id": clip["clip_id"],
                "clip_path": clip["clip_path"],
                "start": clip["start"],
                "end": clip["end"],
                "segments_count": len(rebased_segments),
            })

        merged_segments.sort(key=lambda seg: (seg.get("start", 0.0), seg.get("end", 0.0)))
        merged_segments = self._normalize_segment_ids(merged_segments)

        return {
            "segments": merged_segments,
            "language": detected_language,
            "mode": "vad_clips",
            "clips_expected": len(clip_manifest),
            "clips_processed": len(manifest_with_stats),
            "clip_manifest": manifest_with_stats,
        }

    @staticmethod
    def _normalize_segment_ids(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for index, seg in enumerate(segments):
            item = dict(seg)
            item["id"] = index
            normalized.append(item)
        return normalized

    def _resolve_clip_manifest(self, ctx: StageContext) -> list[dict[str, Any]]:
        if not ctx.vad_segments:
            raise RuntimeError("asr.mode=vad_clips requires VAD segments with exported clips")

        manifest = [
            seg for seg in ctx.vad_segments
            if seg.get("clip_id") and seg.get("clip_path")
        ]
        if not manifest:
            raise RuntimeError("asr.mode=vad_clips requires vad.export_clips=true and clip manifest data")

        return sorted(manifest, key=lambda seg: (float(seg.get("start", 0.0)), seg["clip_id"]))

    @staticmethod
    def _resolve_clip_path(ctx: StageContext, clip: dict[str, Any]) -> Path:
        clip_path = ctx.job_dir / clip["clip_path"]
        if not clip_path.exists():
            raise RuntimeError(f"VAD clip not found: {clip_path}")
        return clip_path

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

        if ctx.config.asr.mode == "vad_clips":
            manifest = ctx.asr_result.get("clip_manifest", []) if ctx.asr_result else []
            manifest_present = len(manifest) > 0
            checks.append(
                CheckResult(
                    name="clip_manifest_present",
                    passed=manifest_present,
                    details=f"Clip manifest entries: {len(manifest)}",
                )
            )
            if not manifest_present:
                all_ok = False

            all_have_clip_id = all(
                seg.get("source_clip_id")
                for seg in ctx.asr_result.get("segments", [])
            ) if ctx.asr_result and ctx.asr_result.get("segments") else False
            checks.append(
                CheckResult(
                    name="segments_have_source_clip_id",
                    passed=all_have_clip_id,
                    details="All merged ASR segments carry source_clip_id.",
                )
            )
            if not all_have_clip_id:
                all_ok = False

        return ValidationResult(ok=all_ok, checks=checks)

    def suggest_fallback(self, attempt_no: int, ctx: StageContext) -> dict[str, Any]:
        """Progressive retry degradation per spec section 5.6:
        1. base params (attempt 1, no changes)
        2. reduce batch_size
        3. switch compute_type (fp16->int8)
        4. downgrade model
        5. fallback to CPU
        """
        asr_cfg = ctx.config.asr
        if attempt_no == 2 and asr_cfg.batch_size > 4:
            asr_cfg.batch_size = max(4, asr_cfg.batch_size // 2)
            return {"action": "reduce_batch_size", "new_batch_size": asr_cfg.batch_size}
        if attempt_no == 3 and asr_cfg.compute_type != "int8":
            old_type = asr_cfg.compute_type
            asr_cfg.compute_type = "int8"
            return {"action": "switch_compute_type", "old": old_type, "new": "int8"}
        if attempt_no == 4 and asr_cfg.model_name != "tiny":
            model_fallback = {"large-v3": "medium", "medium": "small", "small": "base", "base": "tiny"}
            new_model = model_fallback.get(asr_cfg.model_name, "tiny")
            asr_cfg.model_name = new_model
            return {"action": "downgrade_model", "new_model": new_model}
        if attempt_no >= 5 and asr_cfg.device != "cpu":
            asr_cfg.device = "cpu"
            asr_cfg.compute_type = "int8"
            return {"action": "fallback_cpu"}
        return {}
