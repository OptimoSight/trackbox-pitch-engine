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
        # Everything that does not depend on the frame is computed once, not per frame.
        self._lower = np.array(config.lower_hsv, dtype=np.uint8)
        self._upper = np.array(config.upper_hsv, dtype=np.uint8)
        self._scale = config.scale
        self._min_area = config.min_area_px * config.scale**2
        self._max_area_ratio = config.max_area_ratio

    def detect(self, frame: np.ndarray) -> DetectionResult:
        try:
            image = frame
            if self._scale != 1.0:
                image = cv2.resize(
                    frame, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA
                )
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self._lower, self._upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            max_area = self._max_area_ratio * image.shape[0] * image.shape[1]
            best = None
            best_area = 0.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if best_area < area <= max_area:
                    best, best_area = contour, area
            if best is None:
                return DetectionResult.rejected(RejectReason.NO_PITCH)
            if best_area <= self._min_area:
                return DetectionResult.rejected(RejectReason.TOO_SMALL)
            points = best.reshape(-1, 2)
            if len(points) < 3:
                return DetectionResult.rejected(RejectReason.NO_PITCH)
            return DetectionResult.found(Polygon(points / self._scale))
        except Exception:
            return DetectionResult.rejected(RejectReason.NO_PITCH)
