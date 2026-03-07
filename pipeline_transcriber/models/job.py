from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class ExpectedSpeakers(BaseModel):
    min: int = Field(default=1)
    max: int = Field(default=10)


class Job(BaseModel):
    job_id: str
    source_type: Literal["youtube", "local_file"]
    source: str
    language: str = Field(default="auto")
    enable_diarization: bool = Field(default=False)
    enable_word_timestamps: bool = Field(default=True)
    output_formats: list[str] = Field(default_factory=list)
    expected_speakers: ExpectedSpeakers | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_jobs(path: Path) -> list[Job]:
    """Read jobs from JSONL, JSON, or YAML and return validated Job models."""
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        jobs_raw = _load_jsonl_jobs(path)
    elif suffix == ".json":
        jobs_raw = _normalize_jobs_payload(json.loads(path.read_text()))
    elif suffix in {".yaml", ".yml"}:
        jobs_raw = _normalize_jobs_payload(yaml.safe_load(path.read_text()) or [])
    else:
        raise ValueError(f"Unsupported jobs file format: {path.suffix or '<none>'}")

    return [Job.model_validate(raw) for raw in jobs_raw]


def _load_jsonl_jobs(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs


def _normalize_jobs_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    raise ValueError("Jobs file must contain a list of jobs or an object with a 'jobs' list")
