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


class AlignStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.ALIGNMENT

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        if ctx.asr_result is None:
            raise RuntimeError("No ASR result available for alignment")
        if ctx.audio_path is None:
            raise RuntimeError("No audio path available for alignment")

        align_dir = ctx.artifacts_dir / "alignment"
        align_dir.mkdir(parents=True, exist_ok=True)

        align_cfg = ctx.config.alignment
        language = ctx.asr_result.get("language", "en")

        device = self._resolve_device(ctx)

        log.info(
            "alignment_starting",
            language=language,
            device=device,
        )

        try:
            aligned_result = self._run_whisperx_align(ctx, language, device)
        except Exception as exc:
            if align_cfg.allow_fallback_skip and not align_cfg.require_word_alignment:
                log.warning(
                    "alignment_fallback_skip",
                    error=str(exc),
                    reason="Alignment model not available, keeping segment-level timestamps",
                )
                aligned_result = self._fallback_segment_level(ctx)
            else:
                raise

        aligned_segments = aligned_result.get("segments", [])
        all_words = []
        for seg in aligned_segments:
            all_words.extend(seg.get("words", []))

        # Count segments with word-level alignment
        segments_with_words = sum(1 for s in aligned_segments if s.get("words"))
        word_ratio = segments_with_words / max(len(aligned_segments), 1)

        # Save aligned result
        result_path = align_dir / "aligned_result.json"
        result_path.write_text(json.dumps(aligned_result, indent=2, ensure_ascii=False))

        # Save words JSONL
        words_path = align_dir / "words.jsonl"
        with words_path.open("w") as fh:
            for w in all_words:
                fh.write(json.dumps(w, ensure_ascii=False) + "\n")

        # Save report
        report = {
            "language": language,
            "device": device,
            "num_segments": len(aligned_segments),
            "num_words": len(all_words),
            "segments_with_words": segments_with_words,
            "word_alignment_ratio": round(word_ratio, 3),
        }
        report_path = align_dir / "alignment_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        ctx.aligned_result = aligned_result

        artifacts = [str(result_path), str(words_path), str(report_path)]
        log.info(
            "alignment_complete",
            num_words=len(all_words),
            word_alignment_ratio=round(word_ratio, 3),
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={
                "num_words": len(all_words),
                "num_segments": len(aligned_segments),
                "word_alignment_ratio": round(word_ratio, 3),
            },
        )

    def _run_whisperx_align(self, ctx: StageContext, language: str, device: str) -> dict:
        """Run whisperx.align() for word-level timestamps."""
        import whisperx

        align_cfg = ctx.config.alignment
        model_override = align_cfg.align_model_overrides.get(language)

        model, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=model_override,
        )

        audio = whisperx.load_audio(str(ctx.audio_path))
        result = whisperx.align(
            ctx.asr_result["segments"],
            model,
            metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )

        return {
            "segments": result.get("segments", []),
            "language": language,
        }

    def _fallback_segment_level(self, ctx: StageContext) -> dict:
        """Fallback: keep ASR segments without word-level alignment."""
        asr_segments = ctx.asr_result.get("segments", [])
        # Add empty words lists to indicate no word alignment
        segments = [{**seg, "words": []} for seg in asr_segments]
        return {
            "segments": segments,
            "language": ctx.asr_result.get("language", "auto"),
        }

    def _resolve_device(self, ctx: StageContext) -> str:
        device = ctx.config.asr.device
        if device != "auto":
            return device
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

        # Check word alignment ratio against threshold
        if ctx.aligned_result:
            segments = ctx.aligned_result.get("segments", [])
            if segments:
                with_words = sum(1 for s in segments if s.get("words"))
                ratio = with_words / len(segments)
                threshold = ctx.config.qa.min_aligned_words_ratio
                ratio_ok = ratio >= threshold or not ctx.config.alignment.require_word_alignment
                checks.append(
                    CheckResult(
                        name="word_alignment_ratio",
                        passed=ratio_ok,
                        details=f"Word alignment ratio {ratio:.2f} vs threshold {threshold:.2f}",
                    )
                )
                if not ratio_ok:
                    all_ok = False

                # Check that at least one segment has non-empty words
                any_words = any(s.get("words") for s in segments)
                checks.append(
                    CheckResult(
                        name="words_not_all_empty",
                        passed=any_words or not ctx.config.alignment.require_word_alignment,
                        details="At least one segment has word-level data."
                        if any_words
                        else "All segments have empty words lists.",
                    )
                )
                if not any_words and ctx.config.alignment.require_word_alignment:
                    all_ok = False

                # Check word timestamps are within segment bounds (with epsilon)
                epsilon = 0.5  # seconds tolerance
                word_bounds_violations = 0
                for seg in segments:
                    seg_start = seg.get("start", 0)
                    seg_end = seg.get("end", 0)
                    for w in seg.get("words", []):
                        w_start = w.get("start")
                        w_end = w.get("end")
                        if w_start is not None and w_end is not None:
                            if w_start < seg_start - epsilon or w_end > seg_end + epsilon:
                                word_bounds_violations += 1
                word_bounds_ok = word_bounds_violations == 0
                checks.append(
                    CheckResult(
                        name="word_bounds_within_segment",
                        passed=word_bounds_ok,
                        details=f"All words within segment bounds (epsilon={epsilon}s)."
                        if word_bounds_ok
                        else f"{word_bounds_violations} words outside segment bounds.",
                    )
                )
                if not word_bounds_ok:
                    all_ok = False

        return ValidationResult(ok=all_ok, checks=checks)

    def suggest_fallback(self, attempt_no: int, ctx: StageContext) -> dict[str, Any]:
        """Progressive retry degradation per spec section 5.7:
        1. same config (attempt 1, no changes)
        2. try alternate align model if overrides available
        3. reduce alignment scope
        4. fallback skip alignment (if allowed)
        """
        align_cfg = ctx.config.alignment
        if attempt_no == 2:
            # Try alternate align model from overrides if available
            lang = (ctx.asr_result or {}).get("language", "")
            overrides = align_cfg.align_model_overrides
            if lang and lang in overrides:
                return {"action": "use_override_model", "model": overrides[lang]}
            return {}
        if attempt_no == 3:
            # Reduce scope: disable char-level alignment by relaxing threshold
            return {"action": "reduce_alignment_scope"}
        if attempt_no >= 4:
            align_cfg.allow_fallback_skip = True
            return {"action": "enable_fallback_skip"}
        return {}
