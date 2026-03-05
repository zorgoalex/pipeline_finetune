from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class AlertSeverity(str, Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Alert(BaseModel):
    alert_id: str
    job_id: str
    stage: str
    severity: AlertSeverity
    error_code: str
    message: str
    attempts_used: int
    timestamp: datetime
    host: str
    trace_id: str
