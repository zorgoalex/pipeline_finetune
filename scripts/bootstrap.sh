#!/usr/bin/env bash
set -euo pipefail

echo "=== Pipeline Transcriber Bootstrap ==="
echo

# Python
echo "[1/5] Checking Python..."
python3 --version || { echo "ERROR: Python 3.11+ required"; exit 1; }

# ffmpeg
echo "[2/5] Checking ffmpeg..."
if ffmpeg -version 2>/dev/null | head -1; then
    echo "  ffmpeg OK"
else
    echo "  WARNING: ffmpeg not found"
    echo "  Install: sudo apt install ffmpeg  OR  download static build from https://johnvansickle.com/ffmpeg/"
fi

# yt-dlp
echo "[3/5] Checking yt-dlp..."
if yt-dlp --version 2>/dev/null; then
    echo "  yt-dlp OK"
else
    echo "  WARNING: yt-dlp not found. Will install via pip."
fi

# Python deps
echo "[4/5] Installing Python dependencies..."
if command -v uv &>/dev/null; then
    echo "  Using uv..."
    uv sync
    uv pip install -e .
else
    echo "  Using pip..."
    pip install -e .
fi

# Self-check
echo "[5/5] Running self-checks..."
python3 -c "import pipeline_transcriber; print('  pipeline_transcriber OK')"
python3 -c "import pydantic; print(f'  pydantic {pydantic.__version__} OK')"
python3 -c "import structlog; print('  structlog OK')"

echo
echo "=== Bootstrap complete ==="
echo
echo "To install ML dependencies (for real transcription):"
echo "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo "  pip install whisperx pyannote.audio"
echo
echo "For diarization, set HF_TOKEN and run: python3 scripts/check_hf_access.py"
