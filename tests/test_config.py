from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pitch_engine.config import load_config
from pitch_engine.errors import ConfigError

DEFAULT = Path(__file__).resolve().parents[1] / "config" / "default.json"


def _default_data() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(DEFAULT.read_text())
    return data


def _write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


def test_default_config_is_valid() -> None:
    config = load_config(DEFAULT, {})
    assert config.detector.type == "color_threshold"


def test_unknown_key_is_rejected_not_ignored(tmp_path: Path) -> None:
    data = _default_data()
    data["detector"]["sport"] = "football"
    with pytest.raises(ConfigError, match=r"detector\.sport"):
        load_config(_write(tmp_path, data), {})


def test_dead_keys_from_the_prototype_are_rejected(tmp_path: Path) -> None:
    data = _default_data()
    data["confidence_threshold"] = 0.5
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(_write(tmp_path, data), {})


def test_detector_type_is_required(tmp_path: Path) -> None:
    data = _default_data()
    del data["detector"]["type"]
    with pytest.raises(ConfigError, match=r"detector\.type"):
        load_config(_write(tmp_path, data), {})


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("detector", "min_area_px", 0),
        ("detector", "lower_hsv", [90, 40, 40]),  # lower above upper
        ("detector", "upper_hsv", [200, 255, 255]),  # hue is 0..179
        ("detector", "type", "sam_mask_v1"),
        ("video", "path", ""),
    ],
)
def test_bad_values_fail_at_load_time(tmp_path: Path, section: str, key: str, value: Any) -> None:
    data = copy.deepcopy(_default_data())
    data[section][key] = value
    with pytest.raises(ConfigError, match=section):
        load_config(_write(tmp_path, data), {})


def test_all_problems_are_reported_together(tmp_path: Path) -> None:
    data = _default_data()
    data["detector"]["min_area_px"] = -1
    data["video"]["path"] = ""
    with pytest.raises(ConfigError) as info:
        load_config(_write(tmp_path, data), {})
    assert "min_area_px" in str(info.value)
    assert "video.path" in str(info.value)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.json", {})


def test_unparsable_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(path, {})


def test_top_level_must_be_an_object(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(_write(tmp_path, [1, 2]), {})


def test_env_override_is_applied() -> None:
    config = load_config(DEFAULT, {"VIDEO_PATH": "/data/match.mp4"})
    assert config.video.path == "/data/match.mp4"


def test_env_override_is_validated_too() -> None:
    with pytest.raises(ConfigError, match=r"logging\.level"):
        load_config(DEFAULT, {"LOG_LEVEL": "LOUD"})


def test_job_id_is_generated_when_absent_and_overridable() -> None:
    assert len(load_config(DEFAULT, {}).job_id) == 32
    assert load_config(DEFAULT, {"JOB_ID": "abc"}).job_id == "abc"
