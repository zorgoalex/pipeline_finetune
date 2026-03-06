"""CLI entry point for the transcription pipeline."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
import typer

from pipeline_transcriber.models.config import load_config
from pipeline_transcriber.models.job import Job, load_jobs
from pipeline_transcriber.orchestrator import Orchestrator

app = typer.Typer(name="pipeline", help="Transcription + diarization pipeline")


@app.command()
def batch(
    config: Path = typer.Option("config/config.example.yaml", "--config", "-c", help="Config YAML path"),
    jobs_file: Path = typer.Option(..., "--jobs", "-j", help="Jobs JSONL path"),
    resume: bool = typer.Option(False, "--resume", help="Resume from last checkpoint"),
) -> None:
    """Process a batch of jobs from a JSONL file."""
    cfg = load_config(config)
    jobs = load_jobs(jobs_file)
    orch = Orchestrator(cfg)
    report = orch.run_batch(jobs, resume=resume)
    raise typer.Exit(code=0 if report["failed"] == 0 else 1)


@app.command()
def single(
    config: Path = typer.Option("config/config.example.yaml", "--config", "-c"),
    source: str = typer.Argument(..., help="URL or file path"),
    source_type: str = typer.Option("local_file", "--type", "-t", help="youtube | local_file"),
    language: str = typer.Option("auto", "--lang", "-l"),
    diarize: bool = typer.Option(False, "--diarize"),
) -> None:
    """Process a single source."""
    cfg = load_config(config)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job = Job(
        job_id=f"single_{ts}_{uuid.uuid4().hex[:6]}",
        source_type=source_type,
        source=source,
        language=language,
        enable_diarization=diarize,
        enable_word_timestamps=True,
        output_formats=cfg.export.formats,
    )
    orch = Orchestrator(cfg)
    report = orch.run_batch([job])
    raise typer.Exit(code=0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    app()
