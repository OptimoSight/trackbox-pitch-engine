"""Time-based frame sampling on top of ``cv2.VideoCapture``.

Work is proportional to the number of frames *sampled*, not to the length of the file:

* ``seek`` jumps straight to each sampled frame, so frames in between are never decoded
  (beyond the codec's own GOP re-sync).
* ``sequential`` decodes with ``grab()`` and only converts every Nth frame. It still reads the
  whole file, but is cheaper than ``read()`` and works when the frame count is unknown.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from pitch_engine.config import SamplingConfig
from pitch_engine.errors import VideoSourceError

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SampledFrame:
    index: int
    timestamp_s: float
    image: np.ndarray | None  # None: the frame exists but could not be decoded


class VideoFrameSource:
    """Context manager yielding sampled frames plus the metadata needed to interpret them."""

    def __init__(self, path: str, sampling: SamplingConfig) -> None:
        self._path = path
        self._sampling = sampling
        self._cap: cv2.VideoCapture | None = None
        self.strategy = sampling.strategy
        self.fps = 0.0
        self.frame_count = 0  # 0 when the container does not say
        self.width = 0
        self.height = 0
        self.step = 1
        self.start_frame = 0
        self.end_frame: int | None = None
        self.truncated = False  # stream ended before the container's own frame count

    def __enter__(self) -> Self:
        if not Path(self._path).is_file():
            raise VideoSourceError(f"video file not found: {self._path}")
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError(f"could not open video (unsupported or corrupt?): {self._path}")
        self._cap = cap
        try:
            self._read_metadata(cap)
        except VideoSourceError:
            cap.release()
            self._cap = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_metadata(self, cap: cv2.VideoCapture) -> None:
        cfg = self._sampling
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise VideoSourceError(f"video reports an unusable frame rate ({self.fps})")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            raise VideoSourceError(f"video reports an unusable size ({self.width}x{self.height})")
        self.frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.step = max(1, round(self.fps / cfg.sample_fps))
        self.start_frame = round(cfg.start_seconds * self.fps)
        end = self.frame_count or None
        if cfg.duration_seconds is not None:
            window_end = self.start_frame + round(cfg.duration_seconds * self.fps)
            end = window_end if end is None else min(end, window_end)
        self.end_frame = end
        if end is not None and self.start_frame >= end:
            raise VideoSourceError(
                f"nothing to process: start_seconds={cfg.start_seconds} is at or past the "
                f"end of the selected range ({end} frames at {self.fps:.2f} fps)"
            )
        if self.strategy == "seek" and self.end_frame is None:
            log.warning("frame_count_unknown_falling_back_to_sequential")
            self.strategy = "sequential"

    def percent_complete(self, index: int) -> float | None:
        if self.end_frame is None:
            return None
        span = self.end_frame - self.start_frame
        return min(100.0, 100.0 * (index - self.start_frame + self.step) / span)

    def frames(self) -> Iterator[SampledFrame]:
        if self._cap is None:
            raise RuntimeError("VideoFrameSource must be used as a context manager")
        if self.strategy == "seek":
            yield from self._seek_frames(self._cap)
        else:
            yield from self._sequential_frames(self._cap)

    def _seek_frames(self, cap: cv2.VideoCapture) -> Iterator[SampledFrame]:
        assert self.end_frame is not None  # guaranteed by _read_metadata
        for index in range(self.start_frame, self.end_frame, self.step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            yield self._frame(index, ok, image)

    def _sequential_frames(self, cap: cv2.VideoCapture) -> Iterator[SampledFrame]:
        if self.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        index = self.start_frame
        while self.end_frame is None or index < self.end_frame:
            if not cap.grab():
                self.truncated = self.end_frame is not None and index < self.end_frame
                return
            if (index - self.start_frame) % self.step == 0:
                ok, image = cap.retrieve()
                yield self._frame(index, ok, image)
            index += 1

    def _frame(self, index: int, ok: bool, image: np.ndarray | None) -> SampledFrame:
        return SampledFrame(index, index / self.fps, image if ok and image is not None else None)