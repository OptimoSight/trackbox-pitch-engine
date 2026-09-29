"""Command-line interface.

Exit codes: 0 success; 1 unexpected error; 2 bad configuration; 3 video source problem;
4 feed too noisy to trust; 5 job succeeded but the final report could not be delivered;
130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from pitch_engine.config import AppConfig, load_config
from pitch_engine.detectors import build_detector
from pitch_engine.errors import EXIT_REPORT_UNDELIVERED, ConfigError, PitchEngineError
from pitch_engine.logging_setup import configure_logging
from pitch_engine.pipeline import PitchPipeline
from pitch_engine.reporting import build_reporter

log = logging.getLogger(__name__)


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

    log.info("generating_synthetic_video", extra={"video_path": config.video.path})
    generate_synthetic_video(config.video.path)


def _on_sigterm(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(args.config, os.environ)
    except ConfigError as exc:
        # Logging is not configured yet (it depends on the config), so use stderr.
        print(f"configuration error: {exc}", file=sys.stderr)
        return exc.exit_code

    configure_logging(config.logging, config.job_id)
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if args.generate_video:
            _ensure_video(config)
        detector = build_detector(config.detector)
        reporter = build_reporter(config.reporting)
        outcome = PitchPipeline(config, detector, reporter).run()
    except PitchEngineError as exc:
        log.error("run_failed", extra={"error_type": type(exc).__name__, "error": str(exc)})
        return exc.exit_code
    except KeyboardInterrupt:
        log.error("run_interrupted")
        return 130
    except Exception:
        log.exception("run_crashed")
        return 1
    log.info(
        "pipeline_finished",
        extra={
            "valid_detections": outcome.summary.valid_detections,
            "final_report_delivered": outcome.final_report_delivered,
        },
    )
    if not outcome.final_report_delivered:
        # The job itself succeeded; the orchestrator just was not told. Say so with its own code.
        return EXIT_REPORT_UNDELIVERED
    return 0