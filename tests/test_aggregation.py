from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.prepared import prep

from pitch_engine.aggregation import StreamingAggregator
from pitch_engine.models import RejectReason
from pitch_engine.validation import validate_polygon

FRAME_AREA = 1000.0 * 500.0


def test_no_detections_means_no_numbers_not_zeros() -> None:
    assert StreamingAggregator().result() is None


def test_streaming_statistics_match_numpy() -> None:
    rng = np.random.default_rng(0)
    boxes = [
        box(x, y, x + w, y + h)
        for x, y, w, h in zip(
            rng.uniform(0, 100, 50),
            rng.uniform(0, 100, 50),
            rng.uniform(100, 800, 50),
            rng.uniform(100, 380, 50),
            strict=True,
        )
    ]
    aggregator = StreamingAggregator()
    for polygon in boxes:
        aggregator.add(polygon, FRAME_AREA)
    result = aggregator.result()
    ratios = np.array([p.area / FRAME_AREA for p in boxes])
    bounds = np.array([p.bounds for p in boxes])

    assert result is not None
    assert result.valid_detections == 50
    assert result.area_ratio_mean == pytest.approx(ratios.mean())
    assert result.area_ratio_std == pytest.approx(ratios.std())
    assert result.area_ratio_min == pytest.approx(ratios.min())
    assert result.area_ratio_max == pytest.approx(ratios.max())
    assert result.mean_bounds.min_x == pytest.approx(bounds[:, 0].mean())
    assert result.envelope_bounds.max_y == pytest.approx(bounds[:, 3].max())


def test_validation_rejects_bad_geometry() -> None:
    frame = prep(box(0, 0, 100, 100))
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert validate_polygon(bowtie, frame) is RejectReason.INVALID_POLYGON
    assert validate_polygon(box(50, 50, 150, 90), frame) is RejectReason.INVALID_POLYGON
    assert validate_polygon(Polygon(), frame) is RejectReason.INVALID_POLYGON
    assert validate_polygon(box(10, 10, 60, 60), frame) is None