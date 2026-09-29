from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitch_engine.config import ColorThresholdConfig
from pitch_engine.detectors import FieldDetector, build_detector
from pitch_engine.detectors.color_threshold import ColorThresholdDetector
from pitch_engine.models import RejectReason

GREEN = (34, 139, 34)
W, H = 1280, 720


def _green_frame() -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = GREEN
    return frame


def _pitch_frame() -> np.ndarray:
    frame = _green_frame()
    outline = np.array([[100, 100], [1180, 100], [1230, 620], [50, 620]], np.int32)
    cv2.polylines(frame, [outline], True, (255, 255, 255), 5)
    return frame


@pytest.fixture
def detector() -> ColorThresholdDetector:
    return ColorThresholdDetector(ColorThresholdConfig(type="color_threshold"))


def test_detector_satisfies_the_protocol(detector: ColorThresholdDetector) -> None:
    assert isinstance(detector, FieldDetector)
    assert isinstance(build_detector(ColorThresholdConfig(type="color_threshold")), FieldDetector)


def test_finds_the_pitch_region_not_the_whole_frame(detector: ColorThresholdDetector) -> None:
    result = detector.detect(_pitch_frame())
    assert result.polygon is not None
    ratio = result.polygon.area / (W * H)
    assert 0.4 < ratio < 0.7  # the trapezoid, not the 100% frame rectangle
    min_x, min_y, max_x, max_y = result.polygon.bounds
    assert min_x > 40 and min_y > 90 and max_x < 1240 and max_y < 630


def test_black_frame_has_no_pitch(detector: ColorThresholdDetector) -> None:
    result = detector.detect(np.zeros((H, W, 3), dtype=np.uint8))
    assert result.polygon is None
    assert result.reject_reason is RejectReason.NO_PITCH


def test_full_green_closeup_is_not_a_boundary(detector: ColorThresholdDetector) -> None:
    result = detector.detect(_green_frame())
    assert result.polygon is None
    assert result.reject_reason is RejectReason.NO_PITCH


def test_small_noise_is_rejected_as_too_small(detector: ColorThresholdDetector) -> None:
    frame = _green_frame()
    noise = np.array([[10, 10], [40, 10], [40, 30], [10, 30]], np.int32)
    cv2.polylines(frame, [noise], True, (255, 255, 255), 2)
    result = detector.detect(frame)
    assert result.polygon is None
    assert result.reject_reason is RejectReason.TOO_SMALL
