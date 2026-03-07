# Pipeline Transcriber

Automated pipeline for batch video/audio transcription, diarization, and word-level timestamps.

## Stack
- WhisperX (ASR + alignment)
- pyannote.audio (diarization)
- yt-dlp (YouTube download)
- ffmpeg (audio processing)

## Quick Start

```bash
# Install dependencies
./scripts/bootstrap.sh
# or
uv sync

# Run batch processing
pipeline batch --config config/config.example.yaml --jobs jobs/jobs.example.jsonl

# JSON and YAML job lists are also supported
pipeline batch --config config/config.example.yaml --jobs jobs/jobs.example.json
pipeline batch --config config/config.example.yaml --jobs jobs/jobs.example.yaml

# Run single file
pipeline single /path/to/audio.wav

# Resume interrupted batch
pipeline batch --config config/config.example.yaml --jobs jobs/jobs.example.jsonl --resume
```

## Project Structure
- `pipeline_transcriber/` - Main package
  - `main.py` - CLI entry point
  - `orchestrator.py` - Pipeline orchestrator
  - `models/` - Pydantic schemas
  - `stages/` - Pipeline stages
  - `utils/` - Utilities (logging, retry, state, alerts)
- `config/` - Configuration examples
- `jobs/` - Job file examples
- `scripts/` - Bootstrap and check scripts
- `tests/` - Test suite

## Configuration
Copy `config/config.example.yaml` and customize. Key settings:
- `asr.model_name` - Whisper model size
- `asr.device` - cpu/cuda/auto
- `diarization.enabled` - Speaker diarization toggle
- `retry.max_attempts` - Max retries per stage (default: 5)
- `app.max_parallel_jobs` - Number of jobs to run concurrently in batch mode
- `app.resume_enabled` - Global switch for checkpoint resume
- `app.cleanup_policy` - Temp directory cleanup policy (`on_success`, `always`, `never`)

## Execution Semantics
- `FinalizeReportStage` is part of the canonical execution plan, but still runs in a guaranteed finalization contour via `finally`.
- `--resume` can re-run only finalization when `final.json` / `report.json` are missing or corrupt, or after a previous finalizer failure.
- `state.status` tracks the main pipeline outcome; `state.finalization_status` tracks whether finalization completed successfully.
- Finalization failures are recorded explicitly in ledger and stage feedback, without masking the primary pipeline outcome.
- `fail_fast_batch=true` keeps unstarted jobs explicit in `batch_report.json` as `aborted_before_start` instead of omitting them.
- Catastrophic worker-level failures in parallel mode still produce minimal per-job `state.json`, `report.json`, and `final.json`.

## I/O Contract
- Batch job files can be provided as `JSONL`, `JSON`, or `YAML`.
- Export supports `txt`, `srt`, `vtt`, `csv`, and `tsv`, plus mandatory `final.json`.
- Root job directory also carries canonical `segments.jsonl`, and `words.jsonl` when word timings exist.

## Observability
- Structured logs include `event`, `ts`, `host`, `pid`, `thread`, and per-scope fields such as `batch_id`, `job_id`, `stage`, `trace_id`, `attempt`, and `duration_ms` where applicable.
- Alerting is best-effort: alert sink failures are logged, but do not change pipeline/job outcomes or block finalization.
- Alerts are emitted not only for exhausted stage failures, but also for missing diarization secrets, worker/system execution failures, and repeated batch failure streaks.
- Per-job temporary workspaces live under `app.tmp_dir/<batch_id>/<job_id>` and are cleaned according to `app.cleanup_policy`.
- Repair of corrupt `report.json` / `final.json` during finalization leaves a `repair_warnings` trail in the regenerated artifacts.
