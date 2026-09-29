"""Detector-independent sanity checks on a polygon before it may influence any metric."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.prepared import PreparedGeometry

from pitch_engine.models import RejectReason


def validate_polygon(polygon: Polygon, frame_box: PreparedGeometry) -> RejectReason | None:
    """Return why the polygon is unusable, or ``None`` if it is fine.

    Applies to every detector, so a future model that returns a self-intersecting shape or one
    that spills outside the frame is caught here rather than in each implementation.
    """
    if polygon.is_empty or polygon.area <= 0 or not polygon.is_valid:
        return RejectReason.INVALID_POLYGON
    if not frame_box.covers(polygon):
        return RejectReason.INVALID_POLYGON
    return None