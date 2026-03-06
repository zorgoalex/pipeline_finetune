"""Pipeline stages registry."""
from __future__ import annotations

import structlog

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.stages.base import BaseStage
from pipeline_transcriber.stages.input_validate import InputValidateStage
from pipeline_transcriber.stages.download import DownloadStage
from pipeline_transcriber.stages.audio_prepare import AudioPrepareStage
from pipeline_transcriber.stages.vad import VadStage
from pipeline_transcriber.stages.asr import AsrStage
from pipeline_transcriber.stages.align import AlignStage
from pipeline_transcriber.stages.diarize import DiarizeStage
from pipeline_transcriber.stages.assign_speakers import AssignSpeakersStage
from pipeline_transcriber.stages.export import ExportStage
from pipeline_transcriber.stages.qa import QaStage
from pipeline_transcriber.stages.finalize import FinalizeReportStage

logger = structlog.get_logger(__name__)


def build_stage_sequence(config: PipelineConfig, job: Job | None = None) -> list[BaseStage]:
    stages: list[BaseStage] = [
        InputValidateStage(),
        DownloadStage(),
        AudioPrepareStage(),
    ]
    if config.vad.enabled:
        stages.append(VadStage())
    stages.append(AsrStage())

    # Alignment: requires both config enabled AND job requesting word timestamps
    want_alignment = config.alignment.enabled
    if job is not None:
        want_alignment = config.alignment.enabled and job.enable_word_timestamps
        if job.enable_word_timestamps and not config.alignment.enabled:
            logger.warning("alignment_disabled_by_config",
                           job_id=job.job_id, note="job requests word timestamps but config disables alignment")
    if want_alignment:
        stages.append(AlignStage())

    # Diarization: requires both config enabled AND job requesting diarization
    want_diarization = config.diarization.enabled
    if job is not None:
        want_diarization = config.diarization.enabled and job.enable_diarization
        if job.enable_diarization and not config.diarization.enabled:
            logger.warning("diarization_disabled_by_config",
                           job_id=job.job_id, note="job requests diarization but config disables it")
    if want_diarization:
        stages.append(DiarizeStage())
        stages.append(AssignSpeakersStage())

    stages.append(ExportStage())
    stages.append(QaStage())
    # FinalizeReportStage is NOT in the sequence — it runs unconditionally
    # via try/finally in the orchestrator to guarantee report creation.
    return stages
