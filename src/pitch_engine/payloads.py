"""Wire models: exactly what is sent to the reporting service, validated before it leaves."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pitch_engine.errors import PitchEngineError
from pitch_engine.models import RunSummary


def _now() -> datetime:
    return datetime.now(UTC)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProgressReport(_Payload):
    """POST /api/v1/jobs/progress"""

    job_id: str = Field(min_length=1)
    status: Literal["running"] = "running"
    timestamp: datetime = Field(default_factory=_now)
    frames_sampled: int = Field(ge=0)
    valid_detections: int = Field(ge=0)
    rejected_frames: int = Field(ge=0)
    percent_complete: float | None = Field(None, ge=0, le=100)


class ErrorInfo(_Payload):
    type: str
    message: str
    exit_code: int


class EventReport(_Payload):
    """POST /api/v1/jobs/events. ``completed`` carries a summary, ``failed`` carries an error."""

    job_id: str = Field(min_length=1)
    event: Literal["started", "completed", "failed"]
    timestamp: datetime = Field(default_factory=_now)
    summary: RunSummary | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _payload_matches_event(self) -> Self:
        if (self.event == "completed") != (self.summary is not None):
            raise ValueError("summary must be present exactly when event is 'completed'")
        if (self.event == "failed") != (self.error is not None):
            raise ValueError("error must be present exactly when event is 'failed'")
        return self

    @classmethod
    def started(cls, job_id: str) -> EventReport:
        return cls(job_id=job_id, event="started")

    @classmethod
    def completed(cls, job_id: str, summary: RunSummary) -> EventReport:
        return cls(job_id=job_id, event="completed", summary=summary)

    @classmethod
    def failed(cls, job_id: str, exc: BaseException) -> EventReport:
        if isinstance(exc, PitchEngineError):
            exit_code = exc.exit_code
        else:
            exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        error = ErrorInfo(
            type=type(exc).__name__, message=str(exc) or type(exc).__name__, exit_code=exit_code
        )
        return cls(job_id=job_id, event="failed", error=error)
