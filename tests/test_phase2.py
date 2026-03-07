"""Comprehensive unit tests for Phase 2 modules.

Covers:
- utils/subprocess.py
- utils/timecode.py
- utils/ffmpeg.py
- utils/yt_dlp.py
- stages/download.py
- stages/export.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.models.stage import StageStatus
from pipeline_transcriber.stages.base import StageContext
from pipeline_transcriber.stages.download import DownloadStage
from pipeline_transcriber.stages.export import ExportStage
from pipeline_transcriber.utils.ffmpeg import extract_audio, probe_audio
from pipeline_transcriber.utils.subprocess import SubprocessError, run_command
from pipeline_transcriber.utils.timecode import seconds_to_srt, seconds_to_vtt
from pipeline_transcriber.utils.yt_dlp import DownloadError, is_retryable_error


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_context(tmp_path: Path, source_type: str = "local_file", source: str | None = None) -> StageContext:
    """Build a minimal StageContext for testing."""
    if source is None:
        src = tmp_path / "input.wav"
        src.write_bytes(b"RIFF" + b"\x00" * 100)
        source = str(src)

    job = Job(
        job_id="test-phase2",
        source_type=source_type,
        source=source,
    )
    config = PipelineConfig()
    job_dir = tmp_path / "output" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        job=job,
        config=config,
        job_dir=job_dir,
        batch_id="batch-1",
        trace_id="trace-1",
    )


# ===========================================================================
# 1. utils/subprocess.py
# ===========================================================================


class TestRunCommand:
    def test_successful_command(self) -> None:
        rc, stdout, stderr = run_command(["echo", "hello"])
        assert rc == 0
        assert stdout.strip() == "hello"

    def test_failed_command_check_false(self) -> None:
        rc, stdout, stderr = run_command(["false"], check=False)
        assert rc != 0

    def test_failed_command_check_true_raises(self) -> None:
        with pytest.raises(SubprocessError) as exc_info:
            run_command(["false"], check=True)
        assert exc_info.value.returncode != 0
        assert exc_info.value.cmd == ["false"]

    def test_timeout_raises(self) -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            run_command(["sleep", "10"], timeout=1)


class TestSubprocessError:
    def test_attributes(self) -> None:
        err = SubprocessError(1, ["mycmd", "--flag"], "some error output")
        assert err.returncode == 1
        assert err.cmd == ["mycmd", "--flag"]
        assert err.stderr == "some error output"
        assert "mycmd" in str(err)
        assert "1" in str(err)

    def test_stderr_truncated_in_message(self) -> None:
        long_stderr = "x" * 1000
        err = SubprocessError(2, ["cmd"], long_stderr)
        # Message should contain at most 500 chars of stderr
        assert len(str(err)) < len(long_stderr)


# ===========================================================================
# 2. utils/timecode.py
# ===========================================================================


class TestSecondsToSrt:
    def test_zero(self) -> None:
        assert seconds_to_srt(0.0) == "00:00:00,000"

    def test_normal_value(self) -> None:
        assert seconds_to_srt(65.5) == "00:01:05,500"

    def test_large_value(self) -> None:
        assert seconds_to_srt(3661.123) == "01:01:01,123"

    def test_uses_comma(self) -> None:
        result = seconds_to_srt(1.5)
        assert "," in result
        assert "." not in result


class TestSecondsToVtt:
    def test_zero(self) -> None:
        assert seconds_to_vtt(0.0) == "00:00:00.000"

    def test_normal_value(self) -> None:
        assert seconds_to_vtt(65.5) == "00:01:05.500"

    def test_large_value(self) -> None:
        assert seconds_to_vtt(3661.123) == "01:01:01.123"

    def test_uses_dot(self) -> None:
        result = seconds_to_vtt(1.5)
        assert "." in result
        assert "," not in result


class TestTimecodeConsistency:
    """SRT and VTT should produce identical digits, differing only in comma vs dot."""

    @pytest.mark.parametrize("seconds", [0.0, 1.999, 65.5, 3661.123, 7200.0])
    def test_same_digits(self, seconds: float) -> None:
        srt = seconds_to_srt(seconds)
        vtt = seconds_to_vtt(seconds)
        assert srt.replace(",", ".") == vtt


# ===========================================================================
# 3. utils/ffmpeg.py
# ===========================================================================


class TestExtractAudio:
    @patch("pipeline_transcriber.utils.ffmpeg.run_command")
    def test_calls_ffmpeg_with_correct_args(self, mock_run: MagicMock, tmp_path: Path) -> None:
        input_path = tmp_path / "video.mp4"
        output_path = tmp_path / "out" / "audio.wav"
        input_path.write_bytes(b"\x00")
        # Simulate ffmpeg producing an output file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def side_effect(args, **kwargs):
            # Create the output file to pass the exists/size check
            Path(args[-1]).write_bytes(b"\x00" * 100)
            return (0, "", "")

        mock_run.side_effect = side_effect

        result = extract_audio(input_path, output_path, sample_rate=16000, channels=1)

        assert result == output_path
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffmpeg"
        assert "-y" in call_args
        assert "-i" in call_args
        assert str(input_path) in call_args
        assert "-ar" in call_args
        assert "16000" in call_args
        assert "-ac" in call_args
        assert "1" in call_args
        assert str(output_path) in call_args
        assert mock_run.call_args[1].get("check") is True

    @patch("pipeline_transcriber.utils.ffmpeg.run_command")
    def test_normalize_flag(self, mock_run: MagicMock, tmp_path: Path) -> None:
        input_path = tmp_path / "video.mp4"
        output_path = tmp_path / "audio.wav"
        input_path.write_bytes(b"\x00")

        def side_effect(args, **kwargs):
            Path(args[-1]).write_bytes(b"\x00" * 100)
            return (0, "", "")

        mock_run.side_effect = side_effect

        extract_audio(input_path, output_path, normalize=True)
        call_args = mock_run.call_args[0][0]
        assert "-af" in call_args
        assert "loudnorm" in call_args

    @patch("pipeline_transcriber.utils.ffmpeg.run_command")
    def test_raises_on_empty_output(self, mock_run: MagicMock, tmp_path: Path) -> None:
        input_path = tmp_path / "video.mp4"
        output_path = tmp_path / "audio.wav"
        input_path.write_bytes(b"\x00")

        # run_command succeeds but produces no file
        mock_run.return_value = (0, "", "")

        with pytest.raises(RuntimeError, match="empty or missing"):
            extract_audio(input_path, output_path)

    @patch("pipeline_transcriber.utils.ffmpeg.run_command")
    def test_raises_on_zero_size_output(self, mock_run: MagicMock, tmp_path: Path) -> None:
        input_path = tmp_path / "video.mp4"
        output_path = tmp_path / "audio.wav"
        input_path.write_bytes(b"\x00")

        def side_effect(args, **kwargs):
            Path(args[-1]).write_bytes(b"")  # zero-size file
            return (0, "", "")

        mock_run.side_effect = side_effect

        with pytest.raises(RuntimeError, match="empty or missing"):
            extract_audio(input_path, output_path)


class TestProbeAudio:
    @patch("pipeline_transcriber.utils.ffmpeg.run_command")
    def test_parses_ffprobe_json(self, mock_run: MagicMock, tmp_path: Path) -> None:
        ffprobe_output = json.dumps({
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_rate": "44100",
                    "channels": 2,
                    "duration": "120.5",
                }
            ],
            "format": {
                "duration": "120.5",
                "format_name": "wav",
            },
        })
        mock_run.return_value = (0, ffprobe_output, "")

        result = probe_audio(tmp_path / "audio.wav")
        assert result["sample_rate"] == 44100
        assert result["channels"] == 2
        assert result["duration_sec"] == pytest.approx(120.5)
        assert result["codec"] == "pcm_s16le"
        assert result["format_name"] == "wav"

    @patch("pipeline_transcriber.utils.ffmpeg.run_command")
    def test_no_audio_stream_raises(self, mock_run: MagicMock, tmp_path: Path) -> None:
        ffprobe_output = json.dumps({
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
            "format": {"format_name": "mp4"},
        })
        mock_run.return_value = (0, ffprobe_output, "")

        with pytest.raises(RuntimeError, match="No audio stream"):
            probe_audio(tmp_path / "video.mp4")

    @patch("pipeline_transcriber.utils.ffmpeg.run_command")
    def test_probe_passes_correct_args(self, mock_run: MagicMock, tmp_path: Path) -> None:
        ffprobe_output = json.dumps({
            "streams": [{"codec_type": "audio", "sample_rate": "16000", "channels": 1}],
            "format": {},
        })
        mock_run.return_value = (0, ffprobe_output, "")
        audio_file = tmp_path / "test.wav"

        probe_audio(audio_file)

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffprobe"
        assert "-print_format" in call_args
        assert "json" in call_args
        assert "-show_format" in call_args
        assert "-show_streams" in call_args
        assert str(audio_file) in call_args


# ===========================================================================
# 4. utils/yt_dlp.py
# ===========================================================================


class TestDownloadError:
    def test_retryable_default(self) -> None:
        err = DownloadError("something went wrong")
        assert err.retryable is True

    def test_retryable_explicit_false(self) -> None:
        err = DownloadError("Video unavailable", retryable=False)
        assert err.retryable is False

    def test_retryable_explicit_true(self) -> None:
        err = DownloadError("network timeout", retryable=True)
        assert err.retryable is True


class TestIsRetryableError:
    def test_retryable_download_error(self) -> None:
        err = DownloadError("timeout", retryable=True)
        assert is_retryable_error(err) is True

    def test_non_retryable_download_error(self) -> None:
        err = DownloadError("Video unavailable", retryable=False)
        assert is_retryable_error(err) is False

    def test_generic_exception_is_retryable(self) -> None:
        err = RuntimeError("unknown")
        assert is_retryable_error(err) is True

    def test_non_retryable_patterns(self) -> None:
        """Verify _classify_error marks known patterns as non-retryable."""
        from pipeline_transcriber.utils.yt_dlp import _classify_error

        non_retryable_messages = [
            "Video unavailable",
            "Private video",
            "This video has been removed",
            "Sign in to confirm your age",
            "is not a valid URL",
        ]
        for msg in non_retryable_messages:
            exc = SubprocessError(1, ["yt-dlp"], msg)
            classified = _classify_error(exc)
            assert classified.retryable is False, f"Expected non-retryable for: {msg}"

    def test_retryable_pattern(self) -> None:
        from pipeline_transcriber.utils.yt_dlp import _classify_error

        exc = SubprocessError(1, ["yt-dlp"], "HTTP Error 503: Service Unavailable")
        classified = _classify_error(exc)
        assert classified.retryable is True


# ===========================================================================
# 5. stages/download.py
# ===========================================================================


class TestDownloadStageLocalFile:
    def test_local_file_copy(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        stage = DownloadStage()
        result = stage.run(ctx)

        assert result.status == StageStatus.SUCCESS
        assert len(result.artifacts) == 2  # media file + meta json
        assert ctx.download_output_path is not None
        assert ctx.download_output_path.exists()

        # Check metadata was written
        meta_path = ctx.artifacts_dir / "raw" / "source_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["source_type"] == "local_file"
        assert meta["title"] == "input"  # stem of input.wav
        assert meta["ext"] == "wav"

    def test_local_file_not_found(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path, source="/nonexistent/file.wav")
        stage = DownloadStage()
        with pytest.raises(FileNotFoundError, match="Local file not found"):
            stage.run(ctx)

    def test_validate_after_run(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        stage = DownloadStage()
        result = stage.run(ctx)
        validation = stage.validate(ctx, result)
        assert validation.ok is True
        assert validation.ok is True  # next_stage_allowed removed
        assert len(validation.checks) > 0


class TestDownloadStageCanRetry:
    def test_can_retry_none_error(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        stage = DownloadStage()
        assert stage.can_retry(None, ctx) is True

    def test_can_retry_retryable_error(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        stage = DownloadStage()
        err = DownloadError("timeout", retryable=True)
        assert stage.can_retry(err, ctx) is True

    def test_can_retry_non_retryable_error(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        stage = DownloadStage()
        err = DownloadError("Video unavailable", retryable=False)
        assert stage.can_retry(err, ctx) is False

    def test_can_retry_generic_exception(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        stage = DownloadStage()
        err = RuntimeError("something")
        assert stage.can_retry(err, ctx) is True


# ===========================================================================
# 6. stages/export.py
# ===========================================================================


def _segments_fixture() -> list[dict]:
    return [
        {"start": 0.0, "end": 2.5, "text": "Hello world.", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.5, "text": "How are you?", "speaker": "SPEAKER_01"},
    ]


class TestExportStageWithSegments:
    def test_creates_all_format_files(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        stage = ExportStage()
        result = stage.run(ctx)

        assert result.status == StageStatus.SUCCESS
        assert (ctx.job_dir / "final.json").exists()
        assert (ctx.job_dir / "segments.jsonl").exists()
        assert (ctx.job_dir / "transcript.srt").exists()
        assert (ctx.job_dir / "transcript.vtt").exists()
        assert (ctx.job_dir / "transcript.txt").exists()
        assert len(result.artifacts) == 5

    def test_json_content(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        segments = _segments_fixture()
        ctx.asr_result = {"segments": segments}
        ExportStage().run(ctx)

        data = json.loads((ctx.job_dir / "final.json").read_text())
        assert data["segments"] == segments

    def test_txt_content(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        ExportStage().run(ctx)

        txt = (ctx.job_dir / "transcript.txt").read_text()
        assert "[SPEAKER_00] Hello world." in txt
        assert "[SPEAKER_01] How are you?" in txt

    def test_srt_format(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        ExportStage().run(ctx)

        srt = (ctx.job_dir / "transcript.srt").read_text()
        lines = srt.split("\n")
        # First segment: index
        assert lines[0] == "1"
        # Timecodes with comma separator
        assert "00:00:00,000 --> 00:00:02,500" in lines[1]
        # Text with speaker prefix
        assert "[SPEAKER_00] Hello world." in lines[2]
        # Blank separator line
        assert lines[3] == ""
        # Second segment
        assert lines[4] == "2"
        assert "00:00:03,000 --> 00:00:05,500" in lines[5]

    def test_vtt_format(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        ExportStage().run(ctx)

        vtt = (ctx.job_dir / "transcript.vtt").read_text()
        lines = vtt.split("\n")
        # VTT starts with WEBVTT header
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""
        # First cue: timecodes with dot separator
        assert "00:00:00.000 --> 00:00:02.500" in lines[2]
        assert "[SPEAKER_00] Hello world." in lines[3]

    def test_srt_uses_comma_not_dot(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        ExportStage().run(ctx)

        srt = (ctx.job_dir / "transcript.srt").read_text()
        # Timecode lines contain commas
        for line in srt.split("\n"):
            if "-->" in line:
                assert "," in line

    def test_vtt_uses_dot_not_comma(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        ExportStage().run(ctx)

        vtt = (ctx.job_dir / "transcript.vtt").read_text()
        for line in vtt.split("\n"):
            if "-->" in line:
                # Dot in timecodes but no comma
                assert "." in line
                assert "," not in line

    def test_csv_and_tsv_exports(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.job.output_formats = ["csv", "tsv"]
        ctx.asr_result = {"segments": _segments_fixture()}

        result = ExportStage().run(ctx)

        assert result.status == StageStatus.SUCCESS
        csv_path = ctx.job_dir / "transcript.csv"
        tsv_path = ctx.job_dir / "transcript.tsv"
        assert csv_path.exists()
        assert tsv_path.exists()

        csv_lines = csv_path.read_text().splitlines()
        tsv_lines = tsv_path.read_text().splitlines()
        assert csv_lines[0] == "segment_id,start,end,speaker,text,num_words"
        assert tsv_lines[0] == "segment_id\tstart\tend\tspeaker\ttext\tnum_words"
        assert "SPEAKER_00" in csv_lines[1]
        assert "Hello world." in tsv_lines[1]

        final_data = json.loads((ctx.job_dir / "final.json").read_text())
        assert final_data["artifacts"]["csv"] == "transcript.csv"
        assert final_data["artifacts"]["tsv"] == "transcript.tsv"


class TestExportStageEmptySegments:
    def test_empty_segments_creates_files(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        # No asr_result set => defaults to {"segments": []}
        stage = ExportStage()
        result = stage.run(ctx)

        assert result.status == StageStatus.SUCCESS
        assert (ctx.job_dir / "final.json").exists()
        assert (ctx.job_dir / "transcript.srt").exists()
        assert (ctx.job_dir / "transcript.vtt").exists()
        assert (ctx.job_dir / "transcript.txt").exists()

    def test_empty_segments_srt_is_empty_content(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ExportStage().run(ctx)
        srt = (ctx.job_dir / "transcript.srt").read_text()
        # No segment indices
        assert "1" not in srt
        assert "-->" not in srt

    def test_empty_segments_vtt_has_header_only(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ExportStage().run(ctx)
        vtt = (ctx.job_dir / "transcript.vtt").read_text()
        assert vtt.startswith("WEBVTT")
        assert "-->" not in vtt

    def test_empty_segments_txt_is_newline(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ExportStage().run(ctx)
        txt = (ctx.job_dir / "transcript.txt").read_text()
        assert txt == "\n"

    def test_metrics_report_zero_segments(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        result = ExportStage().run(ctx)
        assert result.metrics["num_segments"] == 0


class TestExportStageValidation:
    def test_validate_ok(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        stage = ExportStage()
        result = stage.run(ctx)
        validation = stage.validate(ctx, result)
        assert validation.ok is True
        assert validation.ok is True  # next_stage_allowed removed

    def test_validate_missing_file(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": _segments_fixture()}
        stage = ExportStage()
        result = stage.run(ctx)
        # Remove one file to trigger validation failure
        (ctx.job_dir / "transcript.srt").unlink()
        validation = stage.validate(ctx, result)
        assert validation.ok is False


class TestExportStageFusedPriority:
    """Export should prefer fused_result over aligned_result over asr_result."""

    def test_fused_takes_priority(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": [{"start": 0, "end": 1, "text": "asr"}]}
        ctx.fused_result = {"segments": [{"start": 0, "end": 1, "text": "fused"}]}
        ExportStage().run(ctx)
        data = json.loads((ctx.job_dir / "final.json").read_text())
        assert data["segments"][0]["text"] == "fused"

    def test_aligned_over_asr(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.asr_result = {"segments": [{"start": 0, "end": 1, "text": "asr"}]}
        ctx.aligned_result = {"segments": [{"start": 0, "end": 1, "text": "aligned"}]}
        ExportStage().run(ctx)
        data = json.loads((ctx.job_dir / "final.json").read_text())
        assert data["segments"][0]["text"] == "aligned"
