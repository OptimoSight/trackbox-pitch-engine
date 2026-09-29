"""Green-mask placeholder detector (behaviour identical to the prototype at this stage)."""

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
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > self._config.min_area_px:
                    pts = largest.reshape(-1, 2)
                    if len(pts) >= 3:
                        return DetectionResult.found(Polygon(pts))
        except Exception:
            pass
        return DetectionResult.rejected(RejectReason.NO_PITCH)