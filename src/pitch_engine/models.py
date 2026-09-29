"""Domain data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shapely.geometry import Polygon


class RejectReason(StrEnum):
    """Why a frame produced no usable pitch boundary."""

    NO_PITCH = "no_pitch"
    TOO_SMALL = "too_small"


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
