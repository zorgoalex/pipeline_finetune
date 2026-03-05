Write-Host "=== Pipeline Transcriber Bootstrap ===" -ForegroundColor Cyan
Write-Host "Checking Python..."
python --version
Write-Host "Checking ffmpeg..."
try { ffmpeg -version 2>$null | Select-Object -First 1 } catch { Write-Warning "ffmpeg not found" }
Write-Host "Checking yt-dlp..."
try { yt-dlp --version 2>$null } catch { Write-Warning "yt-dlp not found" }
Write-Host "Installing dependencies..."
if (Get-Command uv -ErrorAction SilentlyContinue) { uv sync } else { pip install -e . }
Write-Host "Self-checks..."
python -c "import pipeline_transcriber; print('OK')"
Write-Host "=== Bootstrap complete ===" -ForegroundColor Green
