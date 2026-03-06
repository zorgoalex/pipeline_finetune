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


class AssignSpeakersStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.SPEAKER_ASSIGNMENT

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        if ctx.diarization_result is None:
            raise RuntimeError("No diarization result available for speaker assignment")

        fusion_dir = ctx.artifacts_dir / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)

        # Use aligned result if available, otherwise raw ASR
        source = ctx.aligned_result or ctx.asr_result
        if source is None:
            raise RuntimeError("No ASR/aligned result available for speaker assignment")

        source_segments = source.get("segments", [])
        diar_segments = ctx.diarization_result.get("segments", [])

        # Try whisperx.assign_word_speakers if available
        try:
            fused_result = self._assign_with_whisperx(ctx, source, diar_segments)
        except Exception:
            log.warning("whisperx_assign_fallback", reason="Using manual overlap-based assignment")
            fused_result = self._assign_manual(source, diar_segments)

        fused_segments = fused_result.get("segments", [])

        # Save fused result
        fused_path = fusion_dir / "fused_result.json"
        fused_path.write_text(json.dumps(fused_result, indent=2, ensure_ascii=False))

        # Compute stats
        assigned_segments = sum(1 for s in fused_segments if s.get("speaker", "UNKNOWN") != "UNKNOWN")
        total_words = sum(len(s.get("words", [])) for s in fused_segments)
        assigned_words = sum(
            sum(1 for w in s.get("words", []) if w.get("speaker", "UNKNOWN") != "UNKNOWN")
            for s in fused_segments
        )

        # Save report
        report = {
            "total_segments": len(fused_segments),
            "assigned_segments": assigned_segments,
            "unassigned_segments": len(fused_segments) - assigned_segments,
            "total_words": total_words,
            "assigned_words": assigned_words,
            "segment_assignment_ratio": round(assigned_segments / max(len(fused_segments), 1), 3),
            "word_assignment_ratio": round(assigned_words / max(total_words, 1), 3),
        }
        report_path = fusion_dir / "fusion_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        ctx.fused_result = fused_result

        artifacts = [str(fused_path), str(report_path)]
        log.info(
            "speaker_assignment_complete",
            assigned_segments=assigned_segments,
            total_segments=len(fused_segments),
            assigned_words=assigned_words,
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={
                "assigned_segments": assigned_segments,
                "segment_assignment_ratio": report["segment_assignment_ratio"],
            },
        )

    def _assign_with_whisperx(
        self, ctx: StageContext, source: dict, diar_segments: list[dict]
    ) -> dict:
        """Use whisperx.assign_word_speakers for precise assignment."""
        import whisperx

        # whisperx expects diarize_df (pandas DataFrame from DiarizationPipeline)
        # We need to reconstruct it from our segments
        import pandas as pd

        rows = []
        for seg in diar_segments:
            rows.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
            })
        diarize_df = pd.DataFrame(rows)

        result = whisperx.assign_word_speakers(diarize_df, source)
        return {
            "segments": result.get("segments", []),
            "language": source.get("language", "auto"),
            "num_speakers": ctx.diarization_result.get("num_speakers", 0),
        }

    def _assign_manual(self, source: dict, diar_segments: list[dict]) -> dict:
        """Manual overlap-based speaker assignment as fallback."""
        source_segments = source.get("segments", [])
        fused_segments = []

        for seg in source_segments:
            speaker = self._find_best_speaker(seg, diar_segments)
            fused_seg = {**seg, "speaker": speaker}

            # Assign speakers to words too
            if "words" in seg:
                fused_words = []
                for word in seg["words"]:
                    word_speaker = self._find_best_speaker(word, diar_segments) if "start" in word else speaker
                    fused_words.append({**word, "speaker": word_speaker})
                fused_seg["words"] = fused_words

            fused_segments.append(fused_seg)

        return {
            "segments": fused_segments,
            "language": source.get("language", "auto"),
        }

    def _find_best_speaker(self, item: dict, diar_segments: list[dict]) -> str:
        """Find the speaker with maximum overlap for a given time interval."""
        item_start = item.get("start", 0)
        item_end = item.get("end", 0)
        if item_start >= item_end:
            return "UNKNOWN"

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for dseg in diar_segments:
            overlap_start = max(item_start, dseg["start"])
            overlap_end = min(item_end, dseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

        return best_speaker

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

        if ctx.fused_result:
            fused_segs = ctx.fused_result.get("segments", [])
            source = ctx.aligned_result or ctx.asr_result or {}
            source_segs = source.get("segments", [])

            # No segments lost
            no_loss = len(fused_segs) >= len(source_segs)
            checks.append(
                CheckResult(
                    name="no_segment_loss",
                    passed=no_loss,
                    details=f"Fused {len(fused_segs)} segments from {len(source_segs)} source segments.",
                )
            )
            if not no_loss:
                all_ok = False

            # Speaker assignment ratio
            if fused_segs:
                assigned = sum(1 for s in fused_segs if s.get("speaker", "UNKNOWN") != "UNKNOWN")
                ratio = assigned / len(fused_segs)
                threshold = ctx.config.qa.min_speaker_assigned_ratio
                ratio_ok = ratio >= threshold
                checks.append(
                    CheckResult(
                        name="speaker_assignment_ratio",
                        passed=ratio_ok,
                        details=f"Assignment ratio {ratio:.2f} vs threshold {threshold:.2f}",
                    )
                )
                if not ratio_ok:
                    all_ok = False

        return ValidationResult(ok=all_ok, checks=checks)
