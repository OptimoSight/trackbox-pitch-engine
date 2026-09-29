"""Pipeline: samples frames from the video and asks a ``FieldDetector`` for a boundary."""

from shapely.geometry import Polygon

from pitch_engine.config import AppConfig
from pitch_engine.detectors import FieldDetector
from pitch_engine.video import VideoFrameSource


class PitchPipeline:
    def __init__(self, config: AppConfig, detector: FieldDetector):
        self._config = config
        self._detector = detector

    def process_video(self):
        video_path = self._config.video.path
        print(f"Starting processing for video: {video_path}")
        detected_polygons = []
        sampled = 0

        with VideoFrameSource(video_path, self._config.sampling) as source:
            for frame in source.frames():
                sampled += 1
                if frame.image is None:
                    continue
                poly = self._detector.detect(frame.image).polygon
                if poly and poly.is_valid:
                    outer_boundary = Polygon(
                        [
                            (0, 0),
                            (source.width, 0),
                            (source.width, source.height),
                            (0, source.height),
                        ]
                    )
                    intersection_area = poly.intersection(outer_boundary).area
                    detected_polygons.append((frame.index, poly, intersection_area))

        print(f"Sampled {sampled} frames. Found {len(detected_polygons)} boundaries.")
        return detected_polygons
