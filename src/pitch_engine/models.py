"""Domain data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon


class RejectReason(StrEnum):
    """Why a frame produced no usable pitch boundary."""

    NO_PITCH = "no_pitch"
    TOO_SMALL = "too_small"
    INVALID_POLYGON = "invalid_polygon"
    UNREADABLE_FRAME = "unreadable_frame"


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Outcome of running a detector on one frame: a polygon, or the reason there is none."""

    polygon: Polygon | None
    reject_reason: RejectReason | None = None

    @classmethod
    def found(cls, polygon: Polygon) -> DetectionResult:
        return cls(polygon=polygon)

    @classmethod
    def rejected(cls, reason: RejectReason) -> DetectionResult:
        return cls(polygon=None, reject_reason=reason)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Bounds(_Model):
    """Axis-aligned box in source-frame pixels. Input for the (future) crop-layout step."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float


class AggregateMetrics(_Model):
    """Statistics over *valid* detections only."""

    valid_detections: int = Field(ge=1)
    area_ratio_mean: float
    area_ratio_std: float
    area_ratio_min: float
    area_ratio_max: float
    mean_bounds: Bounds
    envelope_bounds: Bounds


class RunSummary(_Model):
    """Final, machine-readable result of one run."""

    job_id: str
    video_path: str
    video_fps: float
    video_frame_count: int
    frame_step: int
    sample_strategy: str
    frames_sampled: int = Field(ge=0)
    valid_detections: int = Field(ge=0)
    rejected: dict[str, int]
    invalid_ratio: float = Field(ge=0, le=1)
    aggregate: AggregateMetrics | None
    stream_truncated: bool
    elapsed_seconds: float = Field(ge=0)
    samples_per_second: float = Field(ge=0)
