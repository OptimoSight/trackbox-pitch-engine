"""Pipeline: samples frames from the video and asks a ``FieldDetector`` for a boundary."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from shapely.geometry import Polygon, box
from shapely.prepared import prep

from pitch_engine.config import AppConfig
from pitch_engine.detectors import FieldDetector
from pitch_engine.errors import DetectorError
from pitch_engine.video import VideoFrameSource

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Detection:
    frame_index: int
    polygon: Polygon
    area_ratio: float


@dataclass(frozen=True, slots=True)
class RunSummary:
    frames_sampled: int
    unreadable_frames: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def valid_detections(self) -> int:
        return len(self.detections)


class PitchPipeline:
    def __init__(self, config: AppConfig, detector: FieldDetector) -> None:
        self._config = config
        self._detector = detector

    def run(self) -> RunSummary:
        video_path = self._config.video.path
        detections: list[Detection] = []
        sampled = 0
        unreadable = 0

        with VideoFrameSource(video_path, self._config.sampling) as source:
            log.info(
                "run_started",
                extra={
                    "video_path": video_path,
                    "video_fps": source.fps,
                    "video_frame_count": source.frame_count,
                    "frame_step": source.step,
                    "sample_strategy": source.strategy,
                },
            )
            frame_area = source.width * source.height
            inside_frame = prep(box(0, 0, source.width, source.height))

            for frame in source.frames():
                sampled += 1
                if frame.image is None:
                    unreadable += 1
                    log.warning("frame_unreadable", extra={"frame_index": frame.index})
                    continue
                try:
                    poly = self._detector.detect(frame.image).polygon
                except Exception as exc:
                    # A detector that raises is broken, not "seeing nothing": stop the run.
                    raise DetectorError(f"detector failed on frame {frame.index}: {exc}") from exc
                if poly and poly.is_valid and inside_frame.covers(poly):
                    detections.append(Detection(frame.index, poly, poly.area / frame_area))
                if sampled % self._config.progress_every_samples == 0:
                    log.info(
                        "progress",
                        extra={
                            "frames_sampled": sampled,
                            "boundaries_found": len(detections),
                            "percent_complete": source.percent_complete(frame.index),
                        },
                    )

        log.info(
            "run_finished",
            extra={
                "frames_sampled": sampled,
                "unreadable_frames": unreadable,
                "boundaries_found": len(detections),
            },
        )
        return RunSummary(sampled, unreadable, detections)