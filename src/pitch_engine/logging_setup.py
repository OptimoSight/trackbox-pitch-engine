"""Structured logging: one JSON object per line on stdout, easy for any log shipper to parse.

Log an event with a short snake_case message and put the details in ``extra``; they become
top-level JSON keys, so an operator can filter on ``event``, ``job_id`` or ``frame_index``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from pitch_engine.config import LoggingConfig

_STANDARD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def __init__(self, job_id: str) -> None:
        super().__init__()
        self._job_id = job_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "job_id": self._job_id,
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(config: LoggingConfig, job_id: str) -> None:
    """Replace the root handlers so calling this twice never duplicates output."""
    handler = logging.StreamHandler(sys.stdout)
    if config.format == "json":
        handler.setFormatter(JsonFormatter(job_id))
    else:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s %(levelname)s [{job_id}] %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(config.level)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
