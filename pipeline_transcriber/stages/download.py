from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.utils.ffmpeg import probe_audio
from pipeline_transcriber.utils.yt_dlp import (
    DownloadError,
    download_video,
    is_retryable_error,
)


class DownloadStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.DOWNLOAD

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        raw_dir = ctx.artifacts_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        if ctx.job.source_type == "youtube":
            media_path, meta = self._download_youtube(ctx, raw_dir)
        else:
            media_path, meta = self._copy_local(ctx, raw_dir)

        meta = self._enrich_media_metadata(ctx, media_path, meta)

        meta_path = raw_dir / "source_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        ctx.download_output_path = media_path

        artifacts = [str(media_path), str(meta_path)]
        log.info("download_complete", artifacts=artifacts)
        return StageResult(status=StageStatus.SUCCESS, artifacts=artifacts)

    def _download_youtube(
        self, ctx: StageContext, raw_dir: Path
    ) -> tuple[Path, dict]:
        dl_config = ctx.config.downloader
        downloaded_path, meta = download_video(
            url=ctx.job.source,
            output_dir=raw_dir,
            format=dl_config.format,
            yt_dlp_path=dl_config.yt_dlp_path,
            timeout=dl_config.timeout_sec,
        )
        meta["source_type"] = "youtube"
        return downloaded_path, meta

    def _copy_local(
        self, ctx: StageContext, raw_dir: Path
    ) -> tuple[Path, dict]:
        source = Path(ctx.job.source)
        if not source.exists():
            raise FileNotFoundError(f"Local file not found: {source}")

        dest = raw_dir / source.name
        shutil.copy2(source, dest)
        probe = probe_audio(dest, ffprobe_path=ctx.config.ffmpeg.ffprobe_path)
        meta = {
            "source": str(source),
            "source_type": "local_file",
            "title": source.stem,
            "duration_sec": probe["duration_sec"],
            "ext": source.suffix.lstrip("."),
        }
        return dest, meta

    def _enrich_media_metadata(self, ctx: StageContext, media_path: Path, meta: dict) -> dict:
        probe = probe_audio(media_path, ffprobe_path=ctx.config.ffmpeg.ffprobe_path)
        enriched = dict(meta)
        enriched.update({
            "actual_filename": media_path.name,
            "file_size_bytes": media_path.stat().st_size,
            "sha256": self._compute_sha256(media_path),
            "duration_sec": probe["duration_sec"],
            "probe": probe,
        })
        return enriched

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        checks: list[CheckResult] = []
        all_ok = True
        for artifact in result.artifacts:
            p = Path(artifact)
            exists = p.exists()
            non_empty = exists and p.stat().st_size > 0
            checks.append(
                CheckResult(
                    name=f"file_exists:{p.name}",
                    passed=exists,
                    details=f"{artifact} exists={exists}",
                )
            )
            # Media file must be non-empty (meta file can be small)
            if p.suffix != ".json":
                checks.append(
                    CheckResult(
                        name=f"file_non_empty:{p.name}",
                        passed=non_empty,
                        details=f"{artifact} size={p.stat().st_size if exists else 0}",
                    )
                )
                if not non_empty:
                    all_ok = False
                try:
                    probe = probe_audio(p, ffprobe_path=ctx.config.ffmpeg.ffprobe_path)
                    probe_ok = float(probe["duration_sec"]) > 0
                    checks.append(
                        CheckResult(
                            name=f"media_probe:{p.name}",
                            passed=probe_ok,
                            details=f"duration_sec={probe['duration_sec']}",
                        )
                    )
                    if not probe_ok:
                        all_ok = False
                except Exception as exc:
                    checks.append(
                        CheckResult(
                            name=f"media_probe:{p.name}",
                            passed=False,
                            details=str(exc),
                        )
                    )
                    all_ok = False
            if not exists:
                all_ok = False
        return ValidationResult(ok=all_ok, checks=checks)

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        if error is None:
            return True
        return is_retryable_error(error)
