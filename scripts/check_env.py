#!/usr/bin/env python3
"""Check that all required tools and dependencies are available."""
import shutil
import sys

def check(name, cmd):
    found = shutil.which(cmd)
    status = "OK" if found else "MISSING"
    print(f"  {name}: {status} ({found or 'not in PATH'})")
    return found is not None

print("Environment check:")
ok = True
ok &= check("Python", "python3")
ok &= check("ffmpeg", "ffmpeg")
ok &= check("ffprobe", "ffprobe")
ok &= check("yt-dlp", "yt-dlp")
try:
    import pipeline_transcriber
    print("  pipeline_transcriber: OK")
except ImportError:
    print("  pipeline_transcriber: MISSING (run: uv sync)")
    ok = False
sys.exit(0 if ok else 1)
