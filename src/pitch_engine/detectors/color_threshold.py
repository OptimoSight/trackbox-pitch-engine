"""Green-mask placeholder detector.

Finds the largest green region that is *not* the whole frame. In the synthetic feed the pitch is
a green area enclosed by a white boundary line, so the region we want is the hole in the outer
green ring, not the ring itself. Selecting only the outermost contour (what the prototype did)
returns the frame rectangle on almost every frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

from pitch_engine.config import ColorThresholdConfig
from pitch_engine.models import DetectionResult, RejectReason


class ColorThresholdDetector:
    def __init__(self, config: ColorThresholdConfig) -> None:
        self._config = config

    def detect(self, frame: np.ndarray) -> DetectionResult:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array(self._config.lower_hsv)
            upper = np.array(self._config.upper_hsv)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            max_area = self._config.max_area_ratio * frame.shape[0] * frame.shape[1]
            best = None
            best_area = 0.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if best_area < area <= max_area:
                    best, best_area = contour, area
            if best is None:
                return DetectionResult.rejected(RejectReason.NO_PITCH)
            if best_area <= self._config.min_area_px:
                return DetectionResult.rejected(RejectReason.TOO_SMALL)
            pts = best.reshape(-1, 2)
            if len(pts) >= 3:
                return DetectionResult.found(Polygon(pts))
        except Exception:
            pass
        return DetectionResult.rejected(RejectReason.NO_PITCH)