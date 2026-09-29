"""Pipeline: samples frames from the video and asks a ``FieldDetector`` for a boundary."""

import logging

from shapely.geometry import box
from shapely.prepared import prep

from pitch_engine.config import AppConfig
from pitch_engine.detectors import FieldDetector
from pitch_engine.video import VideoFrameSource

log = logging.getLogger(__name__)


class PitchPipeline:
    def __init__(self, config: AppConfig, detector: FieldDetector):
        self._config = config
        self._detector = detector

    def process_video(self):
        video_path = self._config.video.path
        detected_polygons = []
        sampled = 0

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
            # Invariant for the whole run: build the frame rectangle once, not once per frame.
            frame_box = box(0, 0, source.width, source.height)
            frame_area = frame_box.area
            inside_frame = prep(frame_box)
            for frame in source.frames():
                sampled += 1
                if frame.image is None:
                    log.warning("frame_unreadable", extra={"frame_index": frame.index})
                    continue
                poly = self._detector.detect(frame.image).polygon
                if poly and poly.is_valid and inside_frame.covers(poly):
                    detected_polygons.append((frame.index, poly, poly.area / frame_area))
                if sampled % self._config.progress_every_samples == 0:
                    log.info(
                        "progress",
                        extra={
                            "frames_sampled": sampled,
                            "boundaries_found": len(detected_polygons),
                            "percent_complete": source.percent_complete(frame.index),
                        },
                    )

        log.info(
            "run_finished",
            extra={"frames_sampled": sampled, "boundaries_found": len(detected_polygons)},
        )
        return detected_polygons
