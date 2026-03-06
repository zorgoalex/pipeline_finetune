#!/usr/bin/env python3
"""Check that all required tools and dependencies are available."""
import shutil
import sys


def check_bin(name, cmd):
    found = shutil.which(cmd)
    status = "OK" if found else "MISSING"
    print(f"  {name}: {status} ({found or 'not in PATH'})")
    return found is not None


def check_import(name, module):
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "?")
        print(f"  {name}: OK (v{version})")
        return True
    except ImportError:
        print(f"  {name}: MISSING")
        return False


print("=== Environment Check ===")
print()

print("[System binaries]")
ok = True
ok &= check_bin("Python", "python3")
ok &= check_bin("ffmpeg", "ffmpeg")
ok &= check_bin("ffprobe", "ffprobe")
ok &= check_bin("yt-dlp", "yt-dlp")

print()
print("[Core Python packages]")
ok &= check_import("pipeline_transcriber", "pipeline_transcriber")
ok &= check_import("pydantic", "pydantic")
ok &= check_import("structlog", "structlog")
ok &= check_import("typer", "typer")
ok &= check_import("yaml", "yaml")

print()
print("[ML packages (optional for tests, required for real runs)]")
ml_ok = True
ml_ok &= check_import("torch", "torch")
ml_ok &= check_import("whisperx", "whisperx")
ml_ok &= check_import("pyannote.audio", "pyannote.audio")
ml_ok &= check_import("torchaudio", "torchaudio")

if not ml_ok:
    print()
    print("  ML packages missing. Install with:")
    print("    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("    pip install whisperx pyannote.audio")

print()
if ok:
    print("Core: ALL OK")
else:
    print("Core: SOME MISSING (see above)")

print(f"ML:   {'ALL OK' if ml_ok else 'SOME MISSING (needed for real transcription runs)'}")
sys.exit(0 if ok else 1)
