"""Post-processing stage: punctuation, capitalization, repetitive n-gram filtering."""
from __future__ import annotations

import json
import re
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


class PostProcessStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.POST_PROCESS

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        if ctx.asr_result is None:
            raise RuntimeError("No ASR result available for post-processing")

        pp_cfg = ctx.config.post_process
        segments = ctx.asr_result.get("segments", [])

        stats = {
            "segments_in": len(segments),
            "repetitive_filtered": 0,
            "capitalized": 0,
            "punctuated": 0,
        }

        processed: list[dict[str, Any]] = []
        for seg in segments:
            text = seg.get("text", "")
            original_text = text

            if pp_cfg.filter_repetitive_ngrams:
                text = self._filter_repetitive(text, pp_cfg.max_consecutive_repeats)
                if text != original_text:
                    stats["repetitive_filtered"] += 1

            if pp_cfg.capitalize_sentences:
                text = self._capitalize(text)
                if text != seg.get("text", ""):
                    stats["capitalized"] += 1

            if pp_cfg.add_terminal_punctuation:
                text = self._add_punctuation(text)
                if not original_text.rstrip().endswith((".", "!", "?", "...")):
                    stats["punctuated"] += 1

            new_seg = {**seg, "text": text}
            processed.append(new_seg)

        ctx.asr_result["segments"] = processed
        stats["segments_out"] = len(processed)

        # Save report
        pp_dir = ctx.artifacts_dir / "post_process"
        pp_dir.mkdir(parents=True, exist_ok=True)
        report_path = pp_dir / "post_process_report.json"
        report_path.write_text(json.dumps(stats, indent=2))

        log.info(
            "post_process_complete",
            segments=len(processed),
            repetitive_filtered=stats["repetitive_filtered"],
            capitalized=stats["capitalized"],
            punctuated=stats["punctuated"],
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[str(report_path)],
            metrics=stats,
        )

    @staticmethod
    def _filter_repetitive(text: str, max_repeats: int) -> str:
        """Remove consecutive repeated words/n-grams exceeding max_repeats."""
        if not text.strip():
            return text

        words = text.split()
        if len(words) <= max_repeats:
            return text

        # Single-word repetition
        result: list[str] = []
        repeat_count = 1
        for i, word in enumerate(words):
            if i > 0 and word.lower() == words[i - 1].lower():
                repeat_count += 1
                if repeat_count <= max_repeats:
                    result.append(word)
            else:
                repeat_count = 1
                result.append(word)

        filtered = " ".join(result)

        # Bi-gram repetition (e.g., "көріңі көріңі көріңі көріңі")
        for n in (2, 3):
            filtered = _filter_ngram_repeats(filtered, n, max_repeats)

        return filtered

    @staticmethod
    def _capitalize(text: str) -> str:
        """Capitalize first letter of the text."""
        stripped = text.lstrip()
        if not stripped:
            return text
        leading_space = text[: len(text) - len(stripped)]
        return leading_space + stripped[0].upper() + stripped[1:]

    @staticmethod
    def _add_punctuation(text: str) -> str:
        """Add terminal punctuation if missing."""
        stripped = text.rstrip()
        if not stripped:
            return text
        if stripped[-1] in ".!?":
            return text

        # Simple question detection: ends with interrogative markers
        question_markers = ("ма", "ме", "ба", "бе", "па", "пе", "ғой", "шы", "ші")
        last_word = stripped.split()[-1].lower() if stripped.split() else ""
        if any(last_word.endswith(m) for m in question_markers):
            return stripped + "?"

        return stripped + "."

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        checks: list[CheckResult] = []
        all_ok = True

        # Check report exists
        report_path = ctx.artifacts_dir / "post_process" / "post_process_report.json"
        exists = report_path.exists()
        checks.append(CheckResult(
            name="post_process_report_exists",
            passed=exists,
            details=f"Report exists={exists}",
        ))
        if not exists:
            all_ok = False

        # Check segments still present
        segments = (ctx.asr_result or {}).get("segments", [])
        has_segments = len(segments) > 0
        checks.append(CheckResult(
            name="segments_preserved",
            passed=has_segments,
            details=f"Post-processed segments: {len(segments)}",
        ))

        return ValidationResult(ok=all_ok, checks=checks)

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False


def _filter_ngram_repeats(text: str, n: int, max_repeats: int) -> str:
    """Filter n-gram consecutive repetitions."""
    words = text.split()
    if len(words) < n * 2:
        return text

    result: list[str] = []
    i = 0
    while i < len(words):
        if i + n <= len(words):
            ngram = " ".join(words[i : i + n])
            count = 1
            j = i + n
            while j + n <= len(words) and " ".join(words[j : j + n]).lower() == ngram.lower():
                count += 1
                j += n
            if count > max_repeats:
                # Keep only max_repeats occurrences
                for _ in range(max_repeats):
                    result.extend(words[i : i + n])
                i = j
            else:
                result.append(words[i])
                i += 1
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)
