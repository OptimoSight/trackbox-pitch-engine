"""Streaming aggregation of valid detections.

Constant memory: a two-hour feed does not keep two hours of polygons alive. Only valid
detections are ever added, so rejected frames cannot skew the statistics.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from pitch_engine.models import AggregateMetrics, Bounds


class StreamingAggregator:
    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0  # Welford: running sum of squared deviations
        self._min = math.inf
        self._max = -math.inf
        self._bounds_sum = [0.0, 0.0, 0.0, 0.0]
        self._envelope = [math.inf, math.inf, -math.inf, -math.inf]

    @property
    def count(self) -> int:
        return self._count

    def add(self, polygon: Polygon, frame_area: float) -> None:
        ratio = polygon.area / frame_area
        self._count += 1
        delta = ratio - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (ratio - self._mean)
        self._min = min(self._min, ratio)
        self._max = max(self._max, ratio)

        min_x, min_y, max_x, max_y = polygon.bounds
        for i, value in enumerate((min_x, min_y, max_x, max_y)):
            self._bounds_sum[i] += value
        self._envelope = [
            min(self._envelope[0], min_x),
            min(self._envelope[1], min_y),
            max(self._envelope[2], max_x),
            max(self._envelope[3], max_y),
        ]

    def result(self) -> AggregateMetrics | None:
        """``None`` when nothing valid was seen: no detections means no numbers, never zeros."""
        if self._count == 0:
            return None
        n = self._count
        return AggregateMetrics(
            valid_detections=n,
            area_ratio_mean=self._mean,
            area_ratio_std=math.sqrt(self._m2 / n),
            area_ratio_min=self._min,
            area_ratio_max=self._max,
            mean_bounds=Bounds(
                min_x=self._bounds_sum[0] / n,
                min_y=self._bounds_sum[1] / n,
                max_x=self._bounds_sum[2] / n,
                max_y=self._bounds_sum[3] / n,
            ),
            envelope_bounds=Bounds(
                min_x=self._envelope[0],
                min_y=self._envelope[1],
                max_x=self._envelope[2],
                max_y=self._envelope[3],
            ),
        )
