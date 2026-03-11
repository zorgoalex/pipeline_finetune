"""Comprehensive unit tests for Phase 3 modules.

Covers: rttm utils, VadStage, AlignStage, DiarizeStage, AssignSpeakersStage, QaStage.
All tests run WITHOUT real whisperx, pyannote, torch, or silero installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import StageContext
from pipeline_transcriber.utils.rttm import parse_rttm, write_rttm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_context(tmp_path: Path) -> StageContext:
    job = Job(job_id="test-01", source_type="local_file", source="/tmp/test.wav", output_formats=["json", "txt"])
    config = PipelineConfig()
    job_dir = tmp_path / "output" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(job=job, config=config, job_dir=job_dir, batch_id="b", trace_id="t")


# ===========================================================================
# 1. utils/rttm.py
# ===========================================================================

class TestParseRttm:
    """Tests for parse_rttm()."""

    def test_parse_valid_multi_speaker(self, tmp_path: Path):
        rttm = tmp_path / "test.rttm"
        rttm.write_text(
            "SPEAKER audio 1 0.500 1.200 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER audio 1 2.000 0.800 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
            "SPEAKER audio 1 3.500 2.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        )
        segments = parse_rttm(rttm)
        assert len(segments) == 3
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[0]["start"] == pytest.approx(0.5)
        assert segments[0]["end"] == pytest.approx(0.5 + 1.2)
        assert segments[0]["duration"] == pytest.approx(1.2)
        assert segments[1]["speaker"] == "SPEAKER_01"
        assert segments[2]["speaker"] == "SPEAKER_00"

    def test_parse_empty_file(self, tmp_path: Path):
        rttm = tmp_path / "empty.rttm"
        rttm.write_text("")
        segments = parse_rttm(rttm)
        assert segments == []

    def test_parse_skips_non_speaker_lines(self, tmp_path: Path):
        rttm = tmp_path / "mixed.rttm"
        rttm.write_text(
            "# This is a comment\n"
            "LEXEME audio 1 0.0 1.0 word <NA> <NA> <NA> <NA>\n"
            "SPEAKER audio 1 1.000 2.000 <NA> <NA> SPK1 <NA> <NA>\n"
            "short line\n"
        )
        segments = parse_rttm(rttm)
        assert len(segments) == 1
        assert segments[0]["speaker"] == "SPK1"

    def test_segments_sorted_by_start(self, tmp_path: Path):
        rttm = tmp_path / "unsorted.rttm"
        rttm.write_text(
            "SPEAKER audio 1 5.000 1.000 <NA> <NA> A <NA> <NA>\n"
            "SPEAKER audio 1 1.000 1.000 <NA> <NA> B <NA> <NA>\n"
            "SPEAKER audio 1 3.000 1.000 <NA> <NA> C <NA> <NA>\n"
        )
        segments = parse_rttm(rttm)
        starts = [s["start"] for s in segments]
        assert starts == [1.0, 3.0, 5.0]

    def test_roundtrip_write_then_parse(self, tmp_path: Path):
        original = [
            {"start": 0.5, "end": 1.7, "speaker": "SPK_A"},
            {"start": 2.0, "end": 3.5, "speaker": "SPK_B"},
        ]
        rttm_path = tmp_path / "roundtrip.rttm"
        write_rttm(original, rttm_path)
        parsed = parse_rttm(rttm_path)
        assert len(parsed) == 2
        assert parsed[0]["speaker"] == "SPK_A"
        assert parsed[0]["start"] == pytest.approx(0.5, abs=1e-3)
        assert parsed[0]["end"] == pytest.approx(1.7, abs=1e-3)
        assert parsed[1]["speaker"] == "SPK_B"

    def test_write_rttm_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "output.rttm"
        assert not deep_path.parent.exists()
        write_rttm([{"start": 0.0, "end": 1.0, "speaker": "X"}], deep_path)
        assert deep_path.exists()
        assert deep_path.parent.exists()


# ===========================================================================
# 2. stages/vad.py - VadStage
# ===========================================================================

class TestVadStage:
    """Tests for VadStage (no real VAD backend needed)."""

    def _stage(self):
        from pipeline_transcriber.stages.vad import VadStage
        return VadStage()

    # -- _split_long_segments --

    def test_split_long_segments_splits_above_max(self):
        stage = self._stage()
        segments = [{"start": 0.0, "end": 35.0}]
        result = stage._split_long_segments(segments, max_sec=15.0)
        # 35s / 15s = 3 chunks (15 + 15 + 5)
        assert len(result) == 3
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"] == pytest.approx(15.0)
        assert result[1]["start"] == pytest.approx(15.0)
        assert result[1]["end"] == pytest.approx(30.0)
        assert result[2]["start"] == pytest.approx(30.0)
        assert result[2]["end"] == pytest.approx(35.0)

    def test_split_long_segments_keeps_short(self):
        stage = self._stage()
        segments = [
            {"start": 0.0, "end": 5.0},
            {"start": 6.0, "end": 10.0},
        ]
        result = stage._split_long_segments(segments, max_sec=15.0)
        assert len(result) == 2
        assert result[0] == segments[0]
        assert result[1] == segments[1]

    # -- validate --

    def test_validate_pass_with_valid_segments(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.vad_segments = [
            {"start": 0.0, "end": 1.5},
            {"start": 2.0, "end": 3.0},
        ]
        # Create artifact files so file_exists checks pass
        vad_dir = ctx.artifacts_dir / "vad"
        vad_dir.mkdir(parents=True, exist_ok=True)
        seg_file = vad_dir / "vad_segments.json"
        seg_file.write_text("[]")
        report_file = vad_dir / "vad_report.json"
        report_file.write_text("{}")

        result = StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[str(seg_file), str(report_file)],
        )
        vr = stage.validate(ctx, result)
        assert vr.ok is True
        assert vr.ok is True  # next_stage_allowed removed

    def test_validate_fail_with_empty_segments(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.vad_segments = []
        result = StageResult(status=StageStatus.SUCCESS, artifacts=[])
        vr = stage.validate(ctx, result)
        assert vr.ok is False
        assert vr.retry_recommended is True
        assert "no speech detected" in (vr.retry_reason or "")

    # -- suggest_fallback --

    def test_suggest_fallback_relaxes_after_no_speech_request(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.vad_segments = []
        stage.validate(ctx, StageResult(status=StageStatus.SUCCESS, artifacts=[]))
        orig_speech = ctx.config.vad.min_speech_duration_sec
        orig_silence = ctx.config.vad.min_silence_duration_sec
        fb = stage.suggest_fallback(1, ctx)
        assert fb == {"action": "relax_thresholds"}
        assert ctx.config.vad.min_speech_duration_sec < orig_speech
        assert ctx.config.vad.min_silence_duration_sec < orig_silence

    def test_suggest_fallback_without_no_speech_request_is_noop(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        fb = stage.suggest_fallback(1, ctx)
        assert fb == {}

    def test_validate_passes_terminal_no_speech_after_retry(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.vad_segments = []
        setattr(ctx, "_vad_no_speech_retry_done", True)

        vr = stage.validate(ctx, StageResult(status=StageStatus.SUCCESS, artifacts=[]))

        assert vr.ok is True
        assert any(check.name == "no_speech_detected" and check.passed for check in vr.checks)

    def test_validate_clip_count_mismatch_requests_retry(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.config.vad.export_clips = True
        ctx.vad_segments = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]

        vad_dir = ctx.artifacts_dir / "vad"
        clips_dir = vad_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        (clips_dir / "clip_0000.wav").write_bytes(b"RIFF")

        result = StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[str(clips_dir)],
        )

        vr = stage.validate(ctx, result)

        assert vr.ok is False
        assert vr.retry_recommended is True
        assert "generated clips" in (vr.retry_reason or "")

    # -- stage_name --

    def test_stage_name(self):
        stage = self._stage()
        assert stage.stage_name == StageName.VAD_SEGMENTATION


# ===========================================================================
# 3. stages/align.py - AlignStage
# ===========================================================================

class TestAlignStage:
    """Tests for AlignStage (no real whisperx needed)."""

    def _stage(self):
        from pipeline_transcriber.stages.align import AlignStage
        return AlignStage()

    # -- _fallback_segment_level --

    def test_fallback_segment_level_creates_empty_words(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.asr_result = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "hello"},
                {"start": 2.0, "end": 3.0, "text": "world"},
            ],
            "language": "en",
        }
        result = stage._fallback_segment_level(ctx)
        assert "segments" in result
        assert len(result["segments"]) == 2
        for seg in result["segments"]:
            assert "words" in seg
            assert seg["words"] == []
        assert result["language"] == "en"

    # -- validate --

    def test_validate_pass_with_aligned_result(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.aligned_result = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]},
            ],
        }
        # Create artifact files
        align_dir = ctx.artifacts_dir / "alignment"
        align_dir.mkdir(parents=True, exist_ok=True)
        f1 = align_dir / "aligned_result.json"
        f1.write_text("{}")
        f2 = align_dir / "words.jsonl"
        f2.write_text("")
        f3 = align_dir / "alignment_report.json"
        f3.write_text("{}")

        result = StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[str(f1), str(f2), str(f3)],
        )
        vr = stage.validate(ctx, result)
        assert vr.ok is True

    # -- suggest_fallback --

    def test_suggest_fallback_enables_skip_on_attempt_3(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.config.alignment.allow_fallback_skip = False
        fb = stage.suggest_fallback(3, ctx)
        assert fb == {"action": "enable_fallback_skip"}
        assert ctx.config.alignment.allow_fallback_skip is True

    def test_suggest_fallback_noop_before_attempt_3(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        fb = stage.suggest_fallback(2, ctx)
        assert fb == {}

    # -- stage_name --

    def test_stage_name(self):
        stage = self._stage()
        assert stage.stage_name == StageName.ALIGNMENT


# ===========================================================================
# 4. stages/diarize.py - DiarizeStage
# ===========================================================================

class TestDiarizeStage:
    """Tests for DiarizeStage (no real pyannote needed)."""

    def _stage(self):
        from pipeline_transcriber.stages.diarize import DiarizeStage
        return DiarizeStage()

    # -- can_retry --

    def test_can_retry_hf_token_error_returns_false(self, tmp_path: Path):
        from pipeline_transcriber.stages.diarize import HfTokenError
        stage = self._stage()
        ctx = make_context(tmp_path)
        assert stage.can_retry(HfTokenError("missing token"), ctx) is False

    def test_can_retry_hf_access_error_returns_false(self, tmp_path: Path):
        from pipeline_transcriber.stages.diarize import HfAccessError
        stage = self._stage()
        ctx = make_context(tmp_path)
        assert stage.can_retry(HfAccessError("terms not accepted"), ctx) is False

    def test_can_retry_runtime_error_returns_true(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        assert stage.can_retry(RuntimeError("something broke"), ctx) is True

    def test_classifies_terms_error_as_deterministic_access_failure(self):
        stage = self._stage()
        assert stage._is_deterministic_hf_access_error(
            "You must accept the conditions to access this model"
        ) is True
        assert stage._is_deterministic_hf_access_error(
            "HTTP Error 503: Service Unavailable"
        ) is False

    # -- validate --

    def test_validate_pass_with_segments(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.diarization_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
            ],
            "num_speakers": 2,
        }
        # Create artifact files
        diar_dir = ctx.artifacts_dir / "diarization"
        diar_dir.mkdir(parents=True, exist_ok=True)
        f1 = diar_dir / "diarization_raw.rttm"
        f1.write_text("")
        f2 = diar_dir / "diarization_segments.json"
        f2.write_text("[]")
        f3 = diar_dir / "diarization_report.json"
        f3.write_text("{}")

        result = StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[str(f1), str(f2), str(f3)],
        )
        vr = stage.validate(ctx, result)
        assert vr.ok is True

    def test_validate_fail_without_segments(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.diarization_result = {
            "segments": [],
            "num_speakers": 0,
        }
        result = StageResult(status=StageStatus.SUCCESS, artifacts=[])
        vr = stage.validate(ctx, result)
        assert vr.ok is False

    # -- stage_name --

    def test_stage_name(self):
        stage = self._stage()
        assert stage.stage_name == StageName.SPEAKER_DIARIZATION


# ===========================================================================
# 5. stages/assign_speakers.py - AssignSpeakersStage
# ===========================================================================

class TestAssignSpeakersStage:
    """Tests for AssignSpeakersStage (no real whisperx needed)."""

    def _stage(self):
        from pipeline_transcriber.stages.assign_speakers import AssignSpeakersStage
        return AssignSpeakersStage()

    # -- _find_best_speaker --

    def test_find_best_speaker_max_overlap(self):
        stage = self._stage()
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 1.5, "end": 4.0, "speaker": "B"},
        ]
        # item from 1.0 to 3.0: overlap with A = 1.0s (1.0..2.0), overlap with B = 1.5s (1.5..3.0)
        item = {"start": 1.0, "end": 3.0}
        assert stage._find_best_speaker(item, diar) == "B"

    def test_find_best_speaker_returns_unknown_no_overlap(self):
        stage = self._stage()
        diar = [
            {"start": 5.0, "end": 6.0, "speaker": "A"},
        ]
        item = {"start": 0.0, "end": 1.0}
        assert stage._find_best_speaker(item, diar) == "UNKNOWN"

    def test_find_best_speaker_returns_unknown_zero_duration(self):
        stage = self._stage()
        diar = [{"start": 0.0, "end": 1.0, "speaker": "A"}]
        # start >= end => UNKNOWN
        item = {"start": 0.5, "end": 0.5}
        assert stage._find_best_speaker(item, diar) == "UNKNOWN"

    # -- _assign_manual --

    def test_assign_manual_assigns_speakers_and_words(self):
        stage = self._stage()
        source = {
            "segments": [
                {
                    "start": 0.0, "end": 2.0, "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.8},
                        {"word": "world", "start": 1.0, "end": 1.8},
                    ],
                },
                {
                    "start": 3.0, "end": 5.0, "text": "goodbye",
                    "words": [
                        {"word": "goodbye", "start": 3.0, "end": 4.5},
                    ],
                },
            ],
            "language": "en",
        }
        diar = [
            {"start": 0.0, "end": 2.5, "speaker": "SPK_A"},
            {"start": 2.5, "end": 6.0, "speaker": "SPK_B"},
        ]
        result = stage._assign_manual(source, diar)
        segs = result["segments"]
        assert len(segs) == 2
        assert segs[0]["speaker"] == "SPK_A"
        assert segs[1]["speaker"] == "SPK_B"
        # Words should also have speakers
        assert segs[0]["words"][0]["speaker"] == "SPK_A"
        assert segs[1]["words"][0]["speaker"] == "SPK_B"

    # -- validate --

    def test_validate_no_segment_loss_and_speaker_ratio(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.asr_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "hello"},
            ],
        }
        ctx.fused_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPK_A"},
            ],
        }
        # Create artifact files
        fusion_dir = ctx.artifacts_dir / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)
        f1 = fusion_dir / "fused_result.json"
        f1.write_text("{}")
        f2 = fusion_dir / "fusion_report.json"
        f2.write_text("{}")

        result = StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[str(f1), str(f2)],
        )
        vr = stage.validate(ctx, result)
        assert vr.ok is True
        # Check that both no_segment_loss and speaker_assignment_ratio checks are present
        check_names = [c.name for c in vr.checks]
        assert "no_segment_loss" in check_names
        assert "speaker_assignment_ratio" in check_names

    def test_validate_fails_on_segment_loss(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.asr_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "a"},
                {"start": 1.0, "end": 2.0, "text": "b"},
                {"start": 2.0, "end": 3.0, "text": "c"},
            ],
        }
        ctx.fused_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPK_A"},
            ],
        }
        result = StageResult(status=StageStatus.SUCCESS, artifacts=[])
        vr = stage.validate(ctx, result)
        assert vr.ok is False

    # -- stage_name --

    def test_stage_name(self):
        stage = self._stage()
        assert stage.stage_name == StageName.SPEAKER_ASSIGNMENT


# ===========================================================================
# 6. stages/qa.py - QaStage
# ===========================================================================

class TestQaStage:
    """Tests for QaStage (no external deps needed)."""

    def _stage(self):
        from pipeline_transcriber.stages.qa import QaStage
        return QaStage()

    def test_run_asr_only_no_diarization(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.asr_result = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "hello"},
                {"start": 2.0, "end": 3.0, "text": "world"},
            ],
        }
        # QA checks final.json exists - create it
        final_path = ctx.job_dir / "final.json"
        final_path.write_text("{}")

        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

        # Verify report written
        report_path = ctx.job_dir / "qa_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        check_names = [c["name"] for c in report["checks"]]
        assert "time_intervals_valid" in check_names
        assert "no_negative_durations" in check_names
        # No speaker_assignment_ratio check since no fused_result
        assert "speaker_assignment_ratio" not in check_names

    def test_run_fused_result_has_speaker_ratio_check(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.fused_result = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "hello", "speaker": "SPK_A"},
                {"start": 2.0, "end": 3.0, "text": "world", "speaker": "SPK_B"},
            ],
        }
        final_path = ctx.job_dir / "final.json"
        final_path.write_text("{}")

        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

        report = json.loads((ctx.job_dir / "qa_report.json").read_text())
        check_names = [c["name"] for c in report["checks"]]
        assert "speaker_assignment_ratio" in check_names

    def test_time_intervals_valid_and_no_negative_durations(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.asr_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "ok"},
            ],
        }
        final_path = ctx.job_dir / "final.json"
        final_path.write_text("{}")

        result = stage.run(ctx)
        report = json.loads((ctx.job_dir / "qa_report.json").read_text())
        checks_by_name = {c["name"]: c for c in report["checks"]}
        assert checks_by_name["time_intervals_valid"]["passed"] is True
        assert checks_by_name["no_negative_durations"]["passed"] is True

    def test_validate_ok_when_all_passed(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        from pipeline_transcriber.stages.qa import QA_METRICS_ALL_PASSED, QA_METRICS_CHECKS
        result = StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[],
            metrics={QA_METRICS_ALL_PASSED: True, QA_METRICS_CHECKS: []},
        )
        vr = stage.validate(ctx, result)
        assert vr.ok is True

    def test_validate_fails_on_hard_failure(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        from pipeline_transcriber.stages.qa import QA_METRICS_ALL_PASSED, QA_METRICS_CHECKS
        result = StageResult(
            status=StageStatus.SUCCESS,
            artifacts=[],
            metrics={
                QA_METRICS_ALL_PASSED: False,
                QA_METRICS_CHECKS: [
                    {"name": "segments_non_empty", "passed": False, "details": "0 segments"},
                ],
            },
        )
        vr = stage.validate(ctx, result)
        assert vr.ok is False
        assert vr.retry_recommended is True
        assert vr.retry_target_stage == StageName.ASR_TRANSCRIPTION.value

    def test_run_flags_nan_and_missing_requested_formats(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        ctx.asr_result = {
            "segments": [
                {"start": float("nan"), "end": 1.0, "text": "oops"},
            ],
        }
        (ctx.job_dir / "final.json").write_text("{}")

        result = stage.run(ctx)
        assert result.status == StageStatus.SUCCESS

        report = json.loads((ctx.job_dir / "qa_report.json").read_text())
        checks_by_name = {c["name"]: c for c in report["checks"]}
        assert checks_by_name["no_nan_values"]["passed"] is False
        assert checks_by_name["requested_export_formats_created"]["passed"] is False

    def test_can_retry_returns_false(self, tmp_path: Path):
        stage = self._stage()
        ctx = make_context(tmp_path)
        assert stage.can_retry(RuntimeError("any"), ctx) is False
        assert stage.can_retry(None, ctx) is False

    def test_stage_name(self):
        stage = self._stage()
        assert stage.stage_name == StageName.QA_VALIDATOR
