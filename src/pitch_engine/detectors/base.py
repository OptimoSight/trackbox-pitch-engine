"""The detector seam."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from pitch_engine.models import DetectionResult


@runtime_checkable
class FieldDetector(Protocol):
    """The one point in the pipeline that varies by sport and deployment.

    A detector turns one BGR frame into either a boundary polygon (in source-frame pixel
    coordinates) or a ``RejectReason``. Expected "nothing to see" outcomes are returned as data;
    an exception means a bug or a broken model and stops the run.

    A ``Protocol`` (structural typing) is used rather than an ABC so a new implementation, for
    example a SAM-based one, needs no inheritance and no import from this package.
    """

    def detect(self, frame: np.ndarray) -> DetectionResult: ...
