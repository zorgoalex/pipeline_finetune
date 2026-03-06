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


class VadStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.VAD_SEGMENTATION

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        if ctx.audio_path is None:
            raise RuntimeError("No audio path available for VAD")

        vad_dir = ctx.artifacts_dir / "vad"
        vad_dir.mkdir(parents=True, exist_ok=True)

        vad_cfg = ctx.config.vad
        backend = vad_cfg.backend

        if backend == "silero":
            segments = self._run_silero(ctx)
        elif backend == "whisperx":
            segments = self._run_whisperx_vad(ctx)
        else:
            raise ValueError(f"Unknown VAD backend: {backend}")

        # Post-process: enforce max segment duration
        segments = self._split_long_segments(segments, vad_cfg.max_segment_sec)

        # Save segments
        segments_path = vad_dir / "vad_segments.json"
        segments_path.write_text(json.dumps(segments, indent=2))

        # Save report
        total_speech = sum(s["end"] - s["start"] for s in segments)
        report = {
            "backend": backend,
            "num_segments": len(segments),
            "total_speech_sec": round(total_speech, 3),
            "min_speech_duration": vad_cfg.min_speech_duration_sec,
            "min_silence_duration": vad_cfg.min_silence_duration_sec,
            "max_segment_sec": vad_cfg.max_segment_sec,
        }
        report_path = vad_dir / "vad_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        # Export clips if configured
        artifacts = [str(segments_path), str(report_path)]
        if vad_cfg.export_clips:
            clips_dir = vad_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            self._export_clips(ctx, segments, clips_dir)
            artifacts.append(str(clips_dir))

        ctx.vad_segments = segments

        log.info(
            "vad_complete",
            num_segments=len(segments),
            total_speech_sec=round(total_speech, 3),
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_segments": len(segments), "total_speech_sec": round(total_speech, 3)},
        )

    def _run_silero(self, ctx: StageContext) -> list[dict]:
        """Run Silero VAD on audio file."""
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]

        wav, sr = self._load_audio_tensor(ctx.audio_path)
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=sr,
            min_speech_duration_ms=int(ctx.config.vad.min_speech_duration_sec * 1000),
            min_silence_duration_ms=int(ctx.config.vad.min_silence_duration_sec * 1000),
        )

        segments = []
        padding = ctx.config.vad.padding_sec
        for ts in speech_timestamps:
            start = max(0.0, ts["start"] / sr - padding)
            end = ts["end"] / sr + padding
            segments.append({"start": round(start, 3), "end": round(end, 3)})

        return segments

    def _run_whisperx_vad(self, ctx: StageContext) -> list[dict]:
        """Run WhisperX's built-in VAD (also uses silero internally)."""
        import whisperx

        audio = whisperx.load_audio(str(ctx.audio_path))
        # whisperx.vad.load_vad_model uses silero internally
        vad_model = whisperx.vad.load_vad_model(self._resolve_device(ctx))
        result = whisperx.vad.merge_vad(
            vad_model,
            audio,
            vad_onset=0.5,
            vad_offset=0.363,
        )
        segments = []
        for seg in result:
            segments.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
            })
        return segments

    def _load_audio_tensor(self, audio_path: Path):
        """Load audio as a torch tensor at 16kHz."""
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0), sr

    def _split_long_segments(self, segments: list[dict], max_sec: float) -> list[dict]:
        """Split segments longer than max_sec into chunks."""
        result = []
        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration <= max_sec:
                result.append(seg)
            else:
                start = seg["start"]
                while start < seg["end"]:
                    end = min(start + max_sec, seg["end"])
                    result.append({"start": round(start, 3), "end": round(end, 3)})
                    start = end
        return result

    def _export_clips(self, ctx: StageContext, segments: list[dict], clips_dir: Path) -> None:
        """Export individual WAV clips for each VAD segment using ffmpeg."""
        from pipeline_transcriber.utils.subprocess import run_command

        ffmpeg_path = ctx.config.ffmpeg.ffmpeg_path
        for i, seg in enumerate(segments):
            clip_path = clips_dir / f"clip_{i:04d}.wav"
            args = [
                ffmpeg_path, "-y",
                "-i", str(ctx.audio_path),
                "-ss", str(seg["start"]),
                "-to", str(seg["end"]),
                "-acodec", "pcm_s16le",
                str(clip_path),
            ]
            run_command(args, check=True)

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

        has_segments = ctx.vad_segments is not None and len(ctx.vad_segments) > 0
        checks.append(
            CheckResult(
                name="segments_non_empty",
                passed=has_segments,
                details=f"VAD produced {len(ctx.vad_segments) if ctx.vad_segments else 0} segments.",
            )
        )
        if not has_segments:
            all_ok = False

        # Validate segment ordering and bounds
        if ctx.vad_segments:
            valid_bounds = all(
                0 <= s["start"] < s["end"] for s in ctx.vad_segments
            )
            checks.append(
                CheckResult(
                    name="segment_bounds_valid",
                    passed=valid_bounds,
                    details="All segments have valid start < end >= 0.",
                )
            )
            if not valid_bounds:
                all_ok = False

        return ValidationResult(ok=all_ok, checks=checks)

    def suggest_fallback(self, attempt: int, ctx: StageContext) -> dict[str, Any]:
        """On retry, relax VAD thresholds."""
        vad_cfg = ctx.config.vad
        if attempt == 2:
            vad_cfg.min_speech_duration_sec = max(0.1, vad_cfg.min_speech_duration_sec * 0.5)
            vad_cfg.min_silence_duration_sec = max(0.1, vad_cfg.min_silence_duration_sec * 0.5)
            return {"action": "relax_thresholds"}
        return {}
