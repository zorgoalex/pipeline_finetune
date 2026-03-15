from __future__ import annotations

import json
import os
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
from pipeline_transcriber.utils.rttm import write_rttm


class HfTokenError(Exception):
    """Raised when HuggingFace token is missing or invalid."""
    pass


class HfAccessError(Exception):
    """Raised when Hugging Face model access is deterministically denied."""
    pass


class DiarizeStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.SPEAKER_DIARIZATION

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        if ctx.audio_path is None:
            raise RuntimeError("No audio path available for diarization")

        diar_cfg = ctx.config.diarization
        effective_min, effective_max = self._effective_speaker_bounds(ctx)
        hf_token = os.environ.get(diar_cfg.hf_token_env_var)
        if not hf_token:
            raise HfTokenError(
                f"HuggingFace token not found in env var '{diar_cfg.hf_token_env_var}'. "
                "Required for pyannote diarization. "
                "Set it and accept model conditions at huggingface.co."
            )

        diar_dir = ctx.artifacts_dir / "diarization"
        diar_dir.mkdir(parents=True, exist_ok=True)

        device = self._resolve_device(ctx)

        log.info(
            "diarization_starting",
            pipeline=diar_cfg.pipeline_name,
            device=device,
            min_speakers=effective_min,
            max_speakers=effective_max,
            override_source="job" if ctx.job.expected_speakers else "config",
        )

        try:
            diar_segments, num_speakers = self._run_pyannote(
                ctx, hf_token, device, effective_min, effective_max,
            )
        except Exception as exc:
            if self._is_deterministic_hf_access_error(str(exc)):
                raise HfAccessError(
                    "Hugging Face model access denied for diarization. "
                    "Verify the token and accept the model terms."
                ) from exc
            raise

        # Save RTTM
        rttm_path = diar_dir / "diarization_raw.rttm"
        write_rttm(diar_segments, rttm_path)

        # Save segments JSON
        segments_path = diar_dir / "diarization_segments.json"
        segments_path.write_text(json.dumps(diar_segments, indent=2))

        # Save report
        report = {
            "backend": diar_cfg.backend,
            "pipeline": diar_cfg.pipeline_name,
            "device": device,
            "num_speakers": num_speakers,
            "num_segments": len(diar_segments),
            "min_speakers_config": diar_cfg.min_speakers,
            "max_speakers_config": diar_cfg.max_speakers,
            "min_speakers_effective": effective_min,
            "max_speakers_effective": effective_max,
            "override_source": "job" if ctx.job.expected_speakers else "config",
        }
        report_path = diar_dir / "diarization_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        ctx.diarization_result = {
            "segments": diar_segments,
            "num_speakers": num_speakers,
        }

        artifacts = [str(rttm_path), str(segments_path), str(report_path)]
        log.info(
            "diarization_complete",
            num_speakers=num_speakers,
            num_segments=len(diar_segments),
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_speakers": num_speakers, "num_segments": len(diar_segments)},
        )

    def _run_pyannote(
        self, ctx: StageContext, hf_token: str, device: str,
        min_speakers: int, max_speakers: int,
    ) -> tuple[list[dict], int]:
        """Run pyannote diarization pipeline via whisperx."""
        import whisperx

        diar_cfg = ctx.config.diarization

        diarize_model = whisperx.DiarizationPipeline(
            model_name=diar_cfg.pipeline_name,
            use_auth_token=hf_token,
            device=device,
        )

        diarize_kwargs: dict[str, Any] = {}
        if min_speakers > 0:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers > 0:
            diarize_kwargs["max_speakers"] = max_speakers

        audio = whisperx.load_audio(str(ctx.audio_path))
        diarize_result = diarize_model(audio, **diarize_kwargs)

        # Convert pyannote output to segment dicts
        segments: list[dict] = []
        speakers_seen: set[str] = set()

        for turn, _, speaker in diarize_result.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3),
                "speaker": speaker,
            })
            speakers_seen.add(speaker)

        segments.sort(key=lambda s: s["start"])
        return segments, len(speakers_seen)

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

        if ctx.diarization_result:
            diar_segs = ctx.diarization_result.get("segments", [])
            num_speakers = ctx.diarization_result.get("num_speakers", 0)
            diar_cfg = ctx.config.diarization

            has_segments = len(diar_segs) > 0
            checks.append(
                CheckResult(
                    name="diarization_segments_non_empty",
                    passed=has_segments,
                    details=f"Diarization produced {len(diar_segs)} segments.",
                )
            )
            if not has_segments:
                all_ok = False

            # Validate interval validity: start <= end for all segments
            if diar_segs:
                intervals_valid = all(
                    s.get("start", 0) <= s.get("end", 0) for s in diar_segs
                )
                checks.append(
                    CheckResult(
                        name="diarization_intervals_valid",
                        passed=intervals_valid,
                        details="All diarization intervals have start <= end."
                        if intervals_valid
                        else "Some diarization intervals have start > end.",
                    )
                )
                if not intervals_valid:
                    all_ok = False

            # Check speaker count within effective bounds
            eff_min, eff_max = self._effective_speaker_bounds(ctx)
            speakers_ok = eff_min <= num_speakers <= eff_max
            checks.append(
                CheckResult(
                    name="speaker_count_in_range",
                    passed=speakers_ok,
                    details=f"Found {num_speakers} speakers (expected {eff_min}-{eff_max}).",
                )
            )
            # Speaker count out of range is a warning, not a failure
            if not speakers_ok:
                pass  # informational only

        # Validate RTTM file is non-empty
        rttm_path = ctx.artifacts_dir / "diarization" / "diarization_raw.rttm"
        if rttm_path.exists():
            rttm_non_empty = rttm_path.stat().st_size > 0
            checks.append(
                CheckResult(
                    name="rttm_non_empty",
                    passed=rttm_non_empty,
                    details="RTTM file is non-empty." if rttm_non_empty
                    else "RTTM file exists but is empty.",
                )
            )
            if not rttm_non_empty:
                all_ok = False

        return ValidationResult(ok=all_ok, checks=checks)

    def _effective_speaker_bounds(self, ctx: StageContext) -> tuple[int, int]:
        """Return (min_speakers, max_speakers) using job override if available."""
        diar_cfg = ctx.config.diarization
        if ctx.job.expected_speakers is not None:
            return ctx.job.expected_speakers.min, ctx.job.expected_speakers.max
        return diar_cfg.min_speakers, diar_cfg.max_speakers

    @staticmethod
    def _is_deterministic_hf_access_error(error_message: str) -> bool:
        lowered = error_message.lower()
        patterns = (
            "accept the conditions",
            "accept the user conditions",
            "access to model",
            "gated repo",
            "403 client error",
            "401 client error",
            "repository not found",
            "is not a valid model identifier",
            "cannot access gated repo",
            "authentication error",
        )
        return any(pattern in lowered for pattern in patterns)

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        if isinstance(error, (HfTokenError, HfAccessError)):
            return False
        return True
