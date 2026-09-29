"""Field detectors."""

from pitch_engine.config import ColorThresholdConfig
from pitch_engine.detectors.base import FieldDetector
from pitch_engine.detectors.color_threshold import ColorThresholdDetector
from pitch_engine.errors import ConfigError

# Becomes a union of config classes once a second implementation exists.
DetectorConfig = ColorThresholdConfig


def build_detector(config: DetectorConfig) -> FieldDetector:
    """The only place that maps a detector config to an implementation."""
    if isinstance(config, ColorThresholdConfig):
        return ColorThresholdDetector(config)
    raise ConfigError(f"unsupported detector type: {config.type!r}")


__all__ = ["DetectorConfig", "FieldDetector", "build_detector"]