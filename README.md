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
