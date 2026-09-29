from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from synthetic_generator import generate_synthetic_video

from pitch_engine.config import AppConfig

DEFAULT = Path(__file__).resolve().parents[1] / "config" / "default.json"
FRAMES = 300  # 10 seconds at 30 fps


@pytest.fixture(scope="session")
def video_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("video") / "feed.mp4"
    generate_synthetic_video(str(path), numFrames=FRAMES)
    return str(path)


def config_dict(video_file: str, **overrides: Any) -> dict[str, Any]:
    """The shipped default config pointed at ``video_file``, plus per-section overrides."""
    data: dict[str, Any] = copy.deepcopy(json.loads(DEFAULT.read_text()))
    data["video"]["path"] = video_file
    data["reporting"]["enabled"] = False
    for section, values in overrides.items():
        if isinstance(values, dict):
            data.setdefault(section, {}).update(values)
        else:
            data[section] = values
    return data


@pytest.fixture
def make_config(video_path: str) -> Callable[..., AppConfig]:
    def _make(**overrides: Any) -> AppConfig:
        return AppConfig.model_validate(config_dict(video_path, **overrides))

    return _make


@pytest.fixture
def write_config(tmp_path: Path, video_path: str) -> Callable[..., Path]:
    def _write(**overrides: Any) -> Path:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_dict(video_path, **overrides)))
        return path

    return _write