"""Validated application configuration.

Loading either returns a fully validated ``AppConfig`` or raises ``ConfigError`` naming every
offending key. Nothing is coerced silently and no bad value is replaced by a default.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from pitch_engine.errors import ConfigError

Hue = Annotated[int, Field(ge=0, le=179)]
Channel = Annotated[int, Field(ge=0, le=255)]


class _Strict(BaseModel):
    """Base for every config section: unknown keys are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VideoConfig(_Strict):
    path: str = Field(min_length=1)


class ColorThresholdConfig(_Strict):
    """Green-mask placeholder detector (stands in for a SAM-style model)."""

    type: Literal["color_threshold"]
    lower_hsv: tuple[Hue, Channel, Channel] = (35, 40, 40)
    upper_hsv: tuple[Hue, Channel, Channel] = (85, 255, 255)
    min_area_px: int = Field(1000, ge=1)
    # Regions covering more than this share of the frame are close-ups, not a pitch boundary.
    max_area_ratio: float = Field(0.9, gt=0, le=1)
    # Detect on a downscaled frame (accuracy vs speed); polygons are mapped back to full size.
    scale: float = Field(1.0, gt=0, le=1)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        if any(lo > hi for lo, hi in zip(self.lower_hsv, self.upper_hsv, strict=True)):
            raise ValueError("lower_hsv must not exceed upper_hsv in any channel")
        return self


class SamplingConfig(_Strict):
    """How much of the video is actually looked at."""

    sample_fps: float = Field(1.0, gt=0, le=240)
    # "seek" jumps to each sampled frame; "sequential" decodes everything and keeps every Nth.
    strategy: Literal["seek", "sequential"] = "seek"
    start_seconds: float = Field(0.0, ge=0)
    duration_seconds: float | None = Field(None, gt=0)
    max_consecutive_read_failures: int = Field(5, ge=1)


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"


class QualityConfig(_Strict):
    """When is a run too noisy to trust?"""

    max_invalid_ratio: float = Field(0.5, ge=0, le=1)


class AppConfig(_Strict):
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    progress_every_samples: int = Field(10, ge=1)
    video: VideoConfig
    detector: ColorThresholdConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


# Environment variables that override file values. They go through the same validation.
_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "VIDEO_PATH": ("video", "path"),
    "JOB_ID": ("job_id",),
    "LOG_LEVEL": ("logging", "level"),
}


def load_config(path: Path, env: Mapping[str, str]) -> AppConfig:
    """Read, override from the environment, validate. Raises ``ConfigError`` on any problem."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a JSON object at the top level")

    _apply_env(raw, env)
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_describe(exc)) from exc


def _apply_env(raw: dict[str, Any], env: Mapping[str, str]) -> None:
    for variable, keys in _ENV_OVERRIDES.items():
        value = env.get(variable)
        if value is None:
            continue
        node = raw
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot apply {variable}: '{key}' is not an object")
        node[keys[-1]] = value


def _describe(error: ValidationError) -> str:
    lines = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  - {location}: {item['msg']}")
    return "invalid configuration:\n" + "\n".join(lines)
