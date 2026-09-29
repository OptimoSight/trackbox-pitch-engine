"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pitch_engine.config import AppConfig, load_config
from pitch_engine.detectors import build_detector
from pitch_engine.errors import ConfigError
from pitch_engine.pipeline import PitchPipeline


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pitch-engine", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="generate the synthetic feed first if the configured video file does not exist",
    )
    return parser.parse_args(argv)


def _ensure_video(config: AppConfig) -> None:
    if Path(config.video.path).exists():
        return
    from synthetic_generator import generate_synthetic_video

    generate_synthetic_video(config.video.path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(args.config, os.environ)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return exc.exit_code

    if args.generate_video:
        _ensure_video(config)
    results = PitchPipeline(config, build_detector(config.detector)).process_video()
    print(f"Pipeline finished with {len(results) if results else 0} results.")
    return 0