from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from pipeline_transcriber.models.config import PipelineConfig
from pipeline_transcriber.models.job import Job
from pipeline_transcriber.stages.asr import AsrStage
from pipeline_transcriber.stages.base import StageContext
from pipeline_transcriber.stages.export import ExportStage
from pipeline_transcriber.stages.qa import QaStage


def _make_ctx(tmp_path: Path) -> StageContext:
    job = Job(
        job_id="vad-clips-job",
        source_type="local_file",
        source=str(tmp_path / "input.wav"),
        output_formats=["json", "txt"],
    )
    config = PipelineConfig()
    config.asr.mode = "vad_clips"
    config.vad.enabled = True
    config.vad.export_clips = True
    config.alignment.enabled = False
    config.diarization.enabled = False

    job_dir = tmp_path / "output" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = job_dir / "artifacts" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio_16k_mono.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 64)

    clips_dir = job_dir / "artifacts" / "vad" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "clip_0000.wav").write_bytes(b"RIFF" + b"\x00" * 64)
    (clips_dir / "clip_0001.wav").write_bytes(b"RIFF" + b"\x00" * 64)

    ctx = StageContext(
        job=job,
        config=config,
        job_dir=job_dir,
        batch_id="batch-1",
        trace_id="trace-1",
    )
    ctx.audio_path = audio_path
    ctx.vad_segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "clip_id": "clip_0000",
            "clip_path": "artifacts/vad/clips/clip_0000.wav",
        },
        {
            "start": 20.0,
            "end": 23.0,
            "clip_id": "clip_0001",
            "clip_path": "artifacts/vad/clips/clip_0001.wav",
        },
    ]
    return ctx


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, batch_size=None, language=None):
        results = [
            {
                "language": "ru",
                "segments": [
                    {"start": 0.2, "end": 1.0, "text": "first clip seg a"},
                    {"start": 1.1, "end": 1.8, "text": "first clip seg b"},
                ],
            },
            {
                "language": "ru",
                "segments": [
                    {"start": 0.5, "end": 1.5, "text": "second clip seg"},
                ],
            },
        ]
        result = results[self.calls]
        self.calls += 1
        return result


def _install_fake_whisperx(monkeypatch):
    fake_module = types.SimpleNamespace()
    fake_model = _FakeModel()
    fake_module.load_model = lambda *args, **kwargs: fake_model
    fake_module.load_audio = lambda path: path
    monkeypatch.setitem(sys.modules, "whisperx", fake_module)


def test_asr_vad_clips_rebases_segments_and_tracks_clip_ids(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisperx(monkeypatch)
    ctx = _make_ctx(tmp_path)

    result = AsrStage().run(ctx)

    assert result.status.value == "SUCCESS"
    segments = ctx.asr_result["segments"]
    assert [seg["source_clip_id"] for seg in segments] == [
        "clip_0000", "clip_0000", "clip_0001",
    ]
    assert segments[0]["start"] == 10.2
    assert segments[1]["end"] == 11.8
    assert segments[2]["start"] == 20.5
    assert ctx.asr_result["mode"] == "vad_clips"
    assert ctx.asr_result["clips_expected"] == 2
    assert ctx.asr_result["clips_processed"] == 2
    assert len(ctx.asr_result["clip_manifest"]) == 2

    raw_path = ctx.job_dir / "artifacts" / "asr" / "raw_asr.json"
    raw = json.loads(raw_path.read_text())
    assert raw["mode"] == "vad_clips"
    assert raw["clip_manifest"][0]["clip_id"] == "clip_0000"


def test_asr_vad_clips_validation_requires_clip_manifest_and_source_ids(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisperx(monkeypatch)
    ctx = _make_ctx(tmp_path)

    stage = AsrStage()
    result = stage.run(ctx)
    validation = stage.validate(ctx, result)

    assert validation.ok is True


def test_export_preserves_source_clip_id_in_final_json(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    ctx.asr_result = {
        "segments": [
            {
                "id": 0,
                "start": 10.2,
                "end": 11.0,
                "text": "hello",
                "source_clip_id": "clip_0000",
            }
        ],
        "language": "ru",
        "mode": "vad_clips",
        "clip_manifest": [{"clip_id": "clip_0000", "segments_count": 1}],
    }

    ExportStage().run(ctx)

    final_data = json.loads((ctx.job_dir / "final.json").read_text())
    assert final_data["segments"][0]["source_clip_id"] == "clip_0000"


def test_qa_fails_when_clip_with_asr_output_is_missing_from_merged_segments(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    ctx.asr_result = {
        "segments": [
            {
                "id": 0,
                "start": 10.2,
                "end": 11.0,
                "text": "hello",
                "source_clip_id": "clip_0000",
            }
        ],
        "language": "ru",
        "mode": "vad_clips",
        "clip_manifest": [
            {"clip_id": "clip_0000", "segments_count": 1},
            {"clip_id": "clip_0001", "segments_count": 1},
        ],
    }
    (ctx.job_dir / "final.json").write_text(json.dumps({"job_id": ctx.job.job_id}))

    stage = QaStage()
    result = stage.run(ctx)
    validation = stage.validate(ctx, result)

    assert validation.ok is False
    assert any(check.name == "clip_ids_accounted_for" for check in validation.checks)
