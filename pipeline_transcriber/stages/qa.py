from __future__ import annotations

import json

from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import BaseStage, StageContext

# Metric keys for QA results passed via StageResult.metrics
QA_METRICS_ALL_PASSED = "qa_all_passed"
QA_METRICS_CHECKS = "qa_checks"


class QaStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.QA_VALIDATOR

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        checks: list[dict[str, object]] = []
        warnings: list[str] = []
        qa_cfg = ctx.config.qa

        # 1. Check final.json exists
        final_path = ctx.job_dir / "final.json"
        final_exists = final_path.exists()
        checks.append({
            "name": "final_json_exists",
            "passed": final_exists,
            "details": f"{final_path} exists={final_exists}",
        })

        # 2. Check segments exist
        source = ctx.fused_result or ctx.aligned_result or ctx.asr_result or {}
        segments = source.get("segments", [])
        has_segments = len(segments) > 0
        checks.append({
            "name": "segments_non_empty",
            "passed": has_segments,
            "details": f"Found {len(segments)} segments.",
        })
        if not has_segments:
            warnings.append("No segments found in transcript result.")

        # 3. Validate time intervals
        valid_times = all(
            seg.get("start", 0) < seg.get("end", 0)
            for seg in segments
        ) if segments else True
        checks.append({
            "name": "time_intervals_valid",
            "passed": valid_times,
            "details": "All segments have start < end." if valid_times else "Some segments have invalid time intervals.",
        })

        # 4. Word alignment ratio check (if alignment was run)
        if ctx.aligned_result:
            aligned_segs = ctx.aligned_result.get("segments", [])
            if aligned_segs:
                with_words = sum(1 for s in aligned_segs if s.get("words"))
                ratio = with_words / len(aligned_segs)
                threshold = qa_cfg.min_aligned_words_ratio
                ratio_ok = ratio >= threshold
                checks.append({
                    "name": "word_alignment_ratio",
                    "passed": ratio_ok,
                    "details": f"Word alignment ratio {ratio:.2f} vs threshold {threshold:.2f}",
                })
                if not ratio_ok:
                    if qa_cfg.fail_on_missing_word_timestamps:
                        pass  # will be caught by validate()
                    else:
                        warnings.append(f"Word alignment ratio {ratio:.2f} below threshold {threshold:.2f}")

        # 5. Speaker assignment ratio check (if diarization was run)
        if ctx.fused_result:
            fused_segs = ctx.fused_result.get("segments", [])
            if fused_segs:
                assigned = sum(1 for s in fused_segs if s.get("speaker", "UNKNOWN") != "UNKNOWN")
                ratio = assigned / len(fused_segs)
                threshold = qa_cfg.min_speaker_assigned_ratio
                ratio_ok = ratio >= threshold
                checks.append({
                    "name": "speaker_assignment_ratio",
                    "passed": ratio_ok,
                    "details": f"Speaker assignment ratio {ratio:.2f} vs threshold {threshold:.2f}",
                })
                if not ratio_ok:
                    if qa_cfg.fail_on_missing_diarization:
                        pass  # will be caught by validate()
                    else:
                        warnings.append(f"Speaker assignment ratio {ratio:.2f} below threshold {threshold:.2f}")

        # 6. Check no negative durations
        neg_durations = [
            seg for seg in segments
            if seg.get("end", 0) - seg.get("start", 0) < 0
        ]
        no_neg = len(neg_durations) == 0
        checks.append({
            "name": "no_negative_durations",
            "passed": no_neg,
            "details": f"Found {len(neg_durations)} segments with negative duration." if not no_neg else "No negative durations.",
        })

        all_passed = all(c["passed"] for c in checks)

        report = {
            "job_id": ctx.job.job_id,
            "all_passed": all_passed,
            "checks": checks,
            "warnings": warnings,
        }

        report_path = ctx.job_dir / "qa_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        qa_summary = {
            "job_id": ctx.job.job_id,
            "num_checks": len(checks),
            "num_passed": sum(1 for c in checks if c["passed"]),
            "num_failed": sum(1 for c in checks if not c["passed"]),
            "num_warnings": len(warnings),
        }
        qa_summary_path = ctx.job_dir / "qa_summary.json"
        qa_summary_path.write_text(json.dumps(qa_summary, indent=2))

        artifacts = [str(report_path), str(qa_summary_path)]

        log.info(
            "qa_complete",
            all_passed=all_passed,
            num_checks=len(checks),
            num_warnings=len(warnings),
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            warnings=warnings,
            metrics={
                QA_METRICS_ALL_PASSED: all_passed,
                QA_METRICS_CHECKS: checks,
            },
        )

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        all_passed = result.metrics.get(QA_METRICS_ALL_PASSED, True)
        qa_checks = result.metrics.get(QA_METRICS_CHECKS, [])
        qa_cfg = ctx.config.qa

        if all_passed:
            return ValidationResult(
                ok=True,
                checks=[CheckResult(name="qa_passed", passed=True, details="All QA checks passed.")],
            )

        # Determine which failures are hard vs soft
        failed_checks: list[CheckResult] = []
        has_hard_failure = False

        # Hard failures: unconditional
        hard_fail_names = {"final_json_exists", "segments_non_empty",
                           "time_intervals_valid", "no_negative_durations"}
        # Flag-gated failures
        flag_gated = {
            "word_alignment_ratio": qa_cfg.fail_on_missing_word_timestamps,
            "speaker_assignment_ratio": qa_cfg.fail_on_missing_diarization,
        }

        for check in qa_checks:
            name = check["name"]
            passed = check["passed"]
            if not passed:
                is_hard = name in hard_fail_names
                is_gated_fail = flag_gated.get(name, False)
                if is_hard or is_gated_fail:
                    has_hard_failure = True
                failed_checks.append(CheckResult(
                    name=name,
                    passed=False,
                    details=check.get("details", ""),
                ))

        if has_hard_failure:
            return ValidationResult(
                ok=False,
                checks=failed_checks,
                retry_recommended=False,
                retry_reason="QA checks failed",
            )

        # Soft failures only — pass with warnings
        return ValidationResult(
            ok=True,
            checks=[CheckResult(name="qa_passed_with_warnings", passed=True,
                                details="QA passed; some non-critical checks below threshold.")],
        )

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False
