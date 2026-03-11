from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field


class AppConfig(BaseModel):
    work_dir: Path = Field(default=Path("./output"))
    tmp_dir: Path = Field(default=Path("./tmp"))
    resume_enabled: bool = Field(default=True)
    cleanup_policy: Literal["on_success", "always", "never"] = Field(default="on_success")
    fail_fast_batch: bool = Field(default=False)
    max_parallel_jobs: int = Field(default=1)


class LoggingConfig(BaseModel):
    model_config = {"populate_by_name": True}

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    json_format: bool = Field(default=True, alias="json")
    log_dir: Path = Field(default=Path("./output/logs"))
    file_rotation_mb: int = Field(
        default=10,
        validation_alias=AliasChoices("file_rotation_mb", "file_rotation"),
    )
    retention_count: int = Field(
        default=5,
        validation_alias=AliasChoices("retention_count", "retention"),
    )


class DownloaderConfig(BaseModel):
    enabled: bool = Field(default=True)
    yt_dlp_path: str = Field(default="yt-dlp")
    timeout_sec: int = Field(default=600)
    format: str = Field(default="bestaudio/best")
    retries_internal: int | None = Field(default=None)


class FfmpegConfig(BaseModel):
    ffmpeg_path: str = Field(default="ffmpeg")
    ffprobe_path: str = Field(default="ffprobe")
    audio_sr: int = Field(default=16000)
    audio_channels: int = Field(default=1)
    normalize_audio: bool = Field(default=False)


class VadConfig(BaseModel):
    enabled: bool = Field(default=True)
    backend: Literal["silero", "whisperx", "none"] = Field(default="silero")
    min_speech_duration_sec: float = Field(default=0.3)
    min_silence_duration_sec: float = Field(default=0.3)
    max_segment_sec: float = Field(default=15.0)
    padding_sec: float = Field(default=0.15)
    export_clips: bool = Field(default=False)


class AsrConfig(BaseModel):
    engine: str = Field(default="whisperx")
    model_name: str = Field(default="small")
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto")
    compute_type: Literal["int8", "float16", "float32"] = Field(default="float16")
    beam_size: int = Field(default=5)
    batch_size: int = Field(default=16)
    language: str = Field(default="auto")
    condition_on_previous_text: bool = Field(default=False)
    vad_inside_whisperx: bool = Field(default=False)
    mode: Literal["full_audio", "vad_clips"] = Field(default="full_audio")


class AlignmentConfig(BaseModel):
    enabled: bool = Field(default=True)
    require_word_alignment: bool = Field(default=False)
    allow_fallback_skip: bool = Field(default=True)
    align_model_overrides: dict[str, str] = Field(default_factory=dict)


class DiarizationConfig(BaseModel):
    enabled: bool = Field(default=True)
    backend: str = Field(default="pyannote")
    pipeline_name: str = Field(default="pyannote/speaker-diarization-community-1")
    hf_token_env_var: str = Field(default="HF_TOKEN")
    min_speakers: int = Field(default=1)
    max_speakers: int = Field(default=10)


class ExportConfig(BaseModel):
    formats: list[str] = Field(default=["json", "srt", "vtt", "txt"])
    speaker_prefix: bool = Field(default=True)
    highlight_words: bool = Field(default=False)


class QaConfig(BaseModel):
    min_aligned_words_ratio: float = Field(default=0.7)
    min_speaker_assigned_ratio: float = Field(default=0.9)
    fail_on_missing_word_timestamps: bool = Field(default=False)
    fail_on_missing_diarization: bool = Field(default=True)


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=5)
    backoff_schedule: list[int] = Field(default=[2, 5, 10, 20, 30])


class AlertsConfig(BaseModel):
    enabled: bool = Field(default=True)
    channels: list[str] = Field(default=["stderr", "jsonl"])
    alerts_file: Path = Field(default=Path("./output/alerts.jsonl"))


class PipelineConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    downloader: DownloaderConfig = Field(default_factory=DownloaderConfig)
    ffmpeg: FfmpegConfig = Field(default_factory=FfmpegConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    qa: QaConfig = Field(default_factory=QaConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)


def load_config(path: Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return PipelineConfig.model_validate(raw)
