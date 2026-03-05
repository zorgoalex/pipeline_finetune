"""Pipeline stages registry."""
from __future__ import annotations
from pipeline_transcriber.models.config import PipelineConfig
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


def build_stage_sequence(config: PipelineConfig) -> list[BaseStage]:
    stages: list[BaseStage] = [
        InputValidateStage(),
        DownloadStage(),
        AudioPrepareStage(),
    ]
    if config.vad.enabled:
        stages.append(VadStage())
    stages.append(AsrStage())
    if config.alignment.enabled:
        stages.append(AlignStage())
    if config.diarization.enabled:
        stages.append(DiarizeStage())
        stages.append(AssignSpeakersStage())
    stages.append(ExportStage())
    stages.append(QaStage())
    stages.append(FinalizeReportStage())
    return stages
