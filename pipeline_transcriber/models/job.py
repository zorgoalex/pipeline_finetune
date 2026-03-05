from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

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
    """Read a JSONL file and return a list of validated Job models."""
    jobs: list[Job] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            jobs.append(Job.model_validate(raw))
    return jobs
