#!/usr/bin/env bash
set -euo pipefail
echo "=== Pipeline Transcriber Bootstrap ==="
echo "Checking Python..."
python3 --version || { echo "ERROR: Python 3.11+ required"; exit 1; }
echo "Checking ffmpeg..."
ffmpeg -version 2>/dev/null | head -1 || echo "WARNING: ffmpeg not found. Install with: sudo apt install ffmpeg"
echo "Checking yt-dlp..."
yt-dlp --version 2>/dev/null || echo "WARNING: yt-dlp not found. Install with: pip install yt-dlp"
echo "Installing Python dependencies..."
if command -v uv &>/dev/null; then
    uv sync
else
    pip install -e .
fi
echo "Running self-checks..."
python3 -c "import pipeline_transcriber; print('pipeline_transcriber OK')"
python3 -c "import pydantic; print(f'pydantic {pydantic.__version__} OK')"
python3 -c "import structlog; print('structlog OK')"
echo "=== Bootstrap complete ==="
