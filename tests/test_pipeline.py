from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pitch_engine.config import AppConfig
from pitch_engine.detectors.color_threshold import ColorThresholdDetector
from pitch_engine.errors import DetectorError, FeedQualityError, VideoSourceError
from pitch_engine.models import DetectionResult, RejectReason, RunSummary
from pitch_engine.pipeline import PitchPipeline
from pitch_engine.reporting import NullReporter

MakeConfig = Callable[..., AppConfig]


def _run(config: AppConfig, detector: object) -> RunSummary:
    return PitchPipeline(config, detector, NullReporter()).run().summary  # type: ignore[arg-type]


class _Never:
    def detect(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult.rejected(RejectReason.NO_PITCH)


class _Boom:
    def detect(self, frame: np.ndarray) -> DetectionResult:
        raise ZeroDivisionError("model exploded")


class _FrameEdge:
    """Returns a polygon that spills outside the frame: the pipeline must reject it."""

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult.found(box(-50, 0, 400, 300))


class _Fixed:
    def __init__(self, polygon: Polygon) -> None:
        self._polygon = polygon

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult.found(self._polygon)


def test_dense_run_over_the_synthetic_feed(make_config: MakeConfig) -> None:
    config = make_config(
        sampling={"sample_fps": 30.0, "strategy": "sequential"}, detector={"scale": 0.5}
    )
    summary = _run(config, ColorThresholdDetector(config.detector))

    assert summary.frames_sampled == 300
    assert summary.valid_detections + sum(summary.rejected.values()) == 300
    assert summary.valid_detections >= 270
    assert set(summary.rejected) <= {"no_pitch", "too_small"}
    assert summary.aggregate is not None
    assert 0.4 < summary.aggregate.area_ratio_mean < 0.7  # a real pitch, never the full frame
    assert summary.aggregate.area_ratio_max < 0.9
    assert summary.stream_truncated is False


def test_default_sampling_reads_only_what_it_needs(make_config: MakeConfig) -> None:
    config = make_config()
    summary = _run(config, ColorThresholdDetector(config.detector))
    assert summary.frame_step == 30
    assert summary.frames_sampled == 10
    assert summary.sample_strategy == "seek"


def test_rejected_frames_never_reach_the_metrics(make_config: MakeConfig) -> None:
    config = make_config(quality={"max_invalid_ratio": 1.0})
    summary = _run(config, _Never())
    assert summary.valid_detections == 0
    assert summary.aggregate is None  # no detections -> no numbers, not zeros
    assert summary.rejected == {"no_pitch": 10}


def test_out_of_frame_polygon_is_counted_as_invalid(make_config: MakeConfig) -> None:
    config = make_config(quality={"max_invalid_ratio": 1.0})
    summary = _run(config, _FrameEdge())
    assert summary.rejected == {"invalid_polygon": 10}


def test_valid_polygon_produces_exact_metrics(make_config: MakeConfig) -> None:
    config = make_config()
    summary = _run(config, _Fixed(box(100, 100, 740, 460)))  # 640 x 360 in a 1280 x 720 frame
    assert summary.aggregate is not None
    assert summary.aggregate.area_ratio_mean == pytest.approx(0.25)
    assert summary.aggregate.area_ratio_std == pytest.approx(0.0)


def test_too_many_bad_frames_fails_the_run(make_config: MakeConfig) -> None:
    config = make_config(quality={"max_invalid_ratio": 0.5})
    with pytest.raises(FeedQualityError, match="10 of 10"):
        _run(config, _Never())


def test_detector_exception_is_not_swallowed(make_config: MakeConfig) -> None:
    config = make_config()
    with pytest.raises(DetectorError, match="model exploded") as info:
        _run(config, _Boom())
    assert isinstance(info.value.__cause__, ZeroDivisionError)


def test_missing_video_fails_immediately(make_config: MakeConfig, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = make_config(video={"path": str(tmp_path / "gone.mp4")})
    with pytest.raises(VideoSourceError):
        _run(config, _Never())
