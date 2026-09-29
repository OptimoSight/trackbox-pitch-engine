from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pitch_engine.config import AppConfig, SamplingConfig
from pitch_engine.errors import VideoSourceError
from pitch_engine.video import VideoFrameSource

MakeConfig = Callable[..., AppConfig]


def _indices(config: AppConfig) -> list[int]:
    with VideoFrameSource(config.video.path, config.sampling) as source:
        return [frame.index for frame in source.frames()]


@pytest.mark.parametrize("strategy", ["seek", "sequential"])
def test_samples_one_frame_per_second(make_config: MakeConfig, strategy: str) -> None:
    config = make_config(sampling={"sample_fps": 1.0, "strategy": strategy})
    assert _indices(config) == list(range(0, 300, 30))


def test_strategies_return_identical_pixels(make_config: MakeConfig) -> None:
    images = {}
    for strategy in ("seek", "sequential"):
        config = make_config(sampling={"sample_fps": 1.0, "strategy": strategy})
        with VideoFrameSource(config.video.path, config.sampling) as source:
            images[strategy] = [f.image for f in source.frames()]
    for seek_img, seq_img in zip(images["seek"], images["sequential"], strict=True):
        assert seek_img is not None and seq_img is not None
        assert abs(seek_img.astype(int) - seq_img.astype(int)).mean() < 2


def test_frames_are_full_size_images_with_timestamps(make_config: MakeConfig) -> None:
    config = make_config(sampling={"sample_fps": 2.0})
    with VideoFrameSource(config.video.path, config.sampling) as source:
        frames = list(source.frames())
        assert (source.width, source.height) == (1280, 720)
    assert frames[1].timestamp_s == pytest.approx(0.5)
    assert frames[0].image is not None and frames[0].image.shape == (720, 1280, 3)


def test_window_limits_how_much_is_read(make_config: MakeConfig) -> None:
    config = make_config(sampling={"sample_fps": 1.0, "start_seconds": 2, "duration_seconds": 3})
    assert _indices(config) == [60, 90, 120]


def test_number_of_samples_depends_on_step_not_on_length(make_config: MakeConfig) -> None:
    config = make_config(sampling={"sample_fps": 3.0})
    with VideoFrameSource(config.video.path, config.sampling) as source:
        expected = math.ceil(source.frame_count / source.step)
    assert len(_indices(config)) == expected


def test_missing_file_is_a_video_source_error(make_config: MakeConfig, tmp_path: Path) -> None:
    config = make_config(video={"path": str(tmp_path / "missing.mp4")})
    with (
        pytest.raises(VideoSourceError, match="not found"),
        VideoFrameSource(config.video.path, config.sampling),
    ):
        pass


def test_file_that_is_not_a_video_is_a_video_source_error(
    make_config: MakeConfig, tmp_path: Path
) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    config = make_config(video={"path": str(junk)})
    with pytest.raises(VideoSourceError), VideoFrameSource(config.video.path, config.sampling):
        pass


def test_start_beyond_the_end_is_rejected_up_front(make_config: MakeConfig) -> None:
    config = make_config(sampling={"start_seconds": 60})
    with (
        pytest.raises(VideoSourceError, match="nothing to process"),
        VideoFrameSource(config.video.path, config.sampling),
    ):
        pass


@pytest.mark.parametrize(
    "bad",
    [{"sample_fps": 0}, {"sample_fps": -1}, {"strategy": "random"}, {"duration_seconds": 0}],
)
def test_sampling_config_rejects_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SamplingConfig(**bad)
