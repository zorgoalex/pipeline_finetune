"""Alert manager."""
from __future__ import annotations

import json
import logging
import socket
import sys
from datetime import datetime, timezone
from uuid import uuid4

from pipeline_transcriber.models.alert import Alert, AlertSeverity
from pipeline_transcriber.models.config import AlertsConfig

logger = logging.getLogger(__name__)


class AlertManager:
    """Dispatches alerts to configured channels."""

    def __init__(self, config: AlertsConfig) -> None:
        self.config = config

    def send(
        self,
        job_id: str,
        stage: str,
        severity: AlertSeverity,
        error_code: str,
        message: str,
        attempts_used: int,
        trace_id: str,
    ) -> None:
        """Create an :class:`Alert` and dispatch it to all configured channels."""
        if not self.config.enabled:
            return

        alert = Alert(
            alert_id=str(uuid4()),
            job_id=job_id,
            stage=stage,
            severity=severity,
            error_code=error_code,
            message=message,
            attempts_used=attempts_used,
            timestamp=datetime.now(timezone.utc),
            host=socket.gethostname(),
            trace_id=trace_id,
        )

        alert_json = alert.model_dump_json()

        for channel in self.config.channels:
            try:
                if channel == "stderr":
                    print(alert_json, file=sys.stderr, flush=True)
                elif channel == "jsonl":
                    alerts_file = self.config.alerts_file
                    alerts_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(alerts_file, "a") as fh:
                        fh.write(alert_json + "\n")
            except Exception as exc:  # pragma: no cover - defensive no-throw boundary
                logger.warning(
                    {
                        "event": "alert_dispatch_failed",
                        "job_id": job_id,
                        "stage": stage,
                        "error_code": error_code,
                        "channel": channel,
                        "error": str(exc),
                        "trace_id": trace_id,
                    }
                )
