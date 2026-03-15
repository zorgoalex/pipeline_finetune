"""Real E2E smoke tests per spec section 15.3.

These tests require external dependencies (ffmpeg, whisperx, pyannote)
and are skipped by default. Run with: pytest -m e2e
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def _require_ffmpeg():
    """Skip if ffmpeg is not available."""
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found")


@pytest.fixture
def _require_whisperx():
    """Skip if whisperx is not importable."""
    try:
        import whisperx  # noqa: F401
    except ImportError:
        pytest.skip("whisperx not installed")


@pytest.fixture
def _require_pyannote():
    """Skip if pyannote.audio is not importable."""
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        pytest.skip("pyannote.audio not installed")


class TestE2ELocalFile:
    """Spec 15.3: real E2E with local audio files."""

    @pytest.mark.usefixtures("_require_ffmpeg", "_require_whisperx")
    def test_local_file_without_diarization(self, tmp_path):
        """E2E: local audio file, no diarization."""
        # TODO: Implement when whisperx is available in test env.
        # 1. Create a short WAV file (or use a fixture file)
        # 2. Build config + job with enable_diarization=False
        # 3. Run Orchestrator.run_batch()
        # 4. Assert: final.json exists, QA passed, all requested formats present
        pytest.skip("Awaiting whisperx in test environment")

    @pytest.mark.usefixtures("_require_ffmpeg", "_require_whisperx", "_require_pyannote")
    def test_local_file_with_diarization(self, tmp_path):
        """E2E: local audio file, with diarization."""
        # TODO: Implement when pyannote is available in test env.
        pytest.skip("Awaiting pyannote in test environment")


class TestE2EYouTube:
    """Spec 15.3: real E2E with YouTube URLs (network required)."""

    @pytest.mark.usefixtures("_require_ffmpeg", "_require_whisperx")
    def test_youtube_url(self, tmp_path):
        """E2E: YouTube URL download + transcription."""
        # TODO: Implement with a short public-domain video.
        pytest.skip("Awaiting whisperx in test environment")
